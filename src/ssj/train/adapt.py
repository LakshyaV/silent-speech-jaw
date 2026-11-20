"""Session adaptation: fit a new mounting with a few minutes of enrolment data.

Every time the sensors are put on, their position and coupling differ slightly.
Calibration removes the geometric part; what remains (gain, lever arm, skin
coupling) is absorbed by a per-session affine input adapter. Adapting to a new
session trains *only* that adapter - 23 x 23 + 23 numbers - on a short set of
enrolment prompts while the recogniser stays frozen, which is fast, cannot
forget what the model knows and needs little data. The adapted model is then
exported with the adapter baked in.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.augment import AugmentConfig
from ..data.build import FeatureDataset
from ..data.loader import BucketSampler, UtteranceDataset, collate
from ..models.ctc import MultiResolutionCTC
from .trainer import configure_cpu, load_checkpoint, predict, score_predictions


@dataclass(frozen=True)
class AdaptConfig:
    epochs: int = 40
    batch_size: int = 16
    lr: float = 5e-3
    adapter_l2: float = 1e-2  # pull towards identity: enrolment sets are small
    coarse_weight: float = 0.3
    seed: int = 0
    holdout: float = 0.2  # share of the enrolment utterances used to pick the best epoch


def adapt_session(
    checkpoint: str | Path,
    dataset_dir: str | Path,
    session: str,
    out_checkpoint: str | Path,
    config: AdaptConfig | None = None,
    *,
    report: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Train an input adapter for ``session`` (a ``subject/session`` key of the dataset)."""
    config = config or AdaptConfig()
    say = report or (lambda message: print(message, flush=True))
    configure_cpu()
    torch.manual_seed(config.seed)
    model, payload = load_checkpoint(checkpoint)
    data = FeatureDataset(dataset_dir)
    if data.meta["frontend_fingerprint"] != payload["frontend_fingerprint"]:
        raise ValueError("the enrolment data and the checkpoint were produced by different front-ends")
    items = [i for i in data.items if i.session_key == session]
    if len(items) < 8:
        raise ValueError(f"session {session!r} has only {len(items)} usable utterances; record at least 8")
    rng = np.random.default_rng(config.seed)
    order = rng.permutation(len(items))
    n_held = max(2, round(config.holdout * len(items)))
    held = [items[i] for i in order[:n_held]]
    fit = [items[i] for i in order[n_held:]]

    # Add (or reuse) an adapter slot for this session.
    sessions = list(payload["sessions"])
    if model.adapter_delta is None:
        model.adapter_delta = torch.nn.Parameter(torch.zeros(1, model.n_features, model.n_features))
        model.adapter_bias = torch.nn.Parameter(torch.zeros(1, model.n_features))
    if session not in sessions:
        sessions.append(session)
        mean_delta = (
            model.adapter_delta.data[1:].mean(dim=0, keepdim=True)
            if model.adapter_delta.shape[0] > 1
            else torch.zeros_like(model.adapter_delta.data[:1])
        )
        mean_bias = (
            model.adapter_bias.data[1:].mean(dim=0, keepdim=True)
            if model.adapter_bias.shape[0] > 1
            else torch.zeros_like(model.adapter_bias.data[:1])
        )
        # Start from the average trained adapter: a better prior than identity.
        model.adapter_delta = torch.nn.Parameter(torch.cat((model.adapter_delta.data, mean_delta)))
        model.adapter_bias = torch.nn.Parameter(torch.cat((model.adapter_bias.data, mean_bias)))
        model.n_sessions = len(sessions)
    slot = sessions.index(session) + 1
    index = {session: slot}

    before = score_predictions(predict(model, data, held, {}), held, data.inventory)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("adapter_"))
    optimizer = torch.optim.Adam([model.adapter_delta, model.adapter_bias], lr=config.lr)
    criterion = MultiResolutionCTC(data.inventory, config.coarse_weight, 0.0)
    light = AugmentConfig(
        rate_range=(0.92, 1.1),
        rotation_deg=2.0,
        head_motion_prob=0.3,
        time_masks=0,
        stream_dropout=0.0,
        concat_prob=0.0,
    )
    train_set = UtteranceDataset(data, fit, index, augment_config=light, seed=config.seed)
    loader = DataLoader(
        train_set,
        batch_sampler=BucketSampler(train_set.lengths, config.batch_size, seed=config.seed),
        collate_fn=collate,
    )
    mask = torch.zeros(len(sessions) + 1, 1, 1)
    mask[slot] = 1.0  # only this session's slot may move

    best = {"per": before["per"], "epoch": 0}
    best_state = (model.adapter_delta.data.clone(), model.adapter_bias.data.clone())
    for epoch in range(1, config.epochs + 1):
        model.eval()  # frozen recogniser: no dropout, adapters always applied
        for batch in loader:
            out = model(batch["x"], batch["lengths"], batch["session"])
            loss, _ = criterion(out["logits"], out["lengths"], batch["y"], batch["target_lengths"])
            loss = loss + config.adapter_l2 * (
                model.adapter_delta[slot].pow(2).mean() + model.adapter_bias[slot].pow(2).mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            model.adapter_delta.grad.mul_(mask)
            model.adapter_bias.grad.mul_(mask[:, :, 0])
            optimizer.step()
        scores = score_predictions(predict(model, data, held, index), held, data.inventory)
        if scores["per"] < best["per"] - 1e-4:
            best = {"per": scores["per"], "class_er": scores["class_er"], "epoch": epoch}
            best_state = (model.adapter_delta.data.clone(), model.adapter_bias.data.clone())
        if epoch % 5 == 0 or epoch == 1:
            say(f"epoch {epoch:3d}  held-out PER {scores['per']:.3f}  class ER {scores['class_er']:.3f}")
    model.adapter_delta.data, model.adapter_bias.data = best_state

    payload["model_state"] = model.state_dict()
    payload["model_config"]["session_adapters"] = True
    payload["sessions"] = sessions
    payload.setdefault("adapted", {})[session] = {"before": before, "after": best, "utterances": len(fit)}
    torch.save(payload, out_checkpoint)
    say(
        f"held-out PER {before['per']:.3f} -> {best['per']:.3f} (best epoch {best['epoch']}, {len(fit)} enrolment utterances)"
    )
    return {
        "session": session,
        "before": before,
        "after": best,
        "enrolment_utterances": len(fit),
        "held_out": len(held),
    }
