"""Training loop for the phoneme recogniser.

Model selection is by phoneme error rate on the validation split, which the
dataset builder makes a held-out *session* whenever it can. Every run writes its
configuration, a per-epoch history and a self-describing checkpoint (weights,
inventory, normalisation and the front-end fingerprint they belong to).
"""

from __future__ import annotations

import csv
import logging
import math
import os
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import from_dict, save_json, save_yaml, to_dict
from ..data.augment import AugmentConfig
from ..data.build import FeatureDataset, Item
from ..data.loader import BucketSampler, UtteranceDataset, collate
from ..decode.ctc_score import greedy_decode, marginalize
from ..eval.metrics import ErrorRate
from ..models.ctc import MultiResolutionCTC
from ..models.encoder import JawPhonemeNet, ModelConfig
from ..phonology import Inventory

log = logging.getLogger(__name__)
CHECKPOINT_FORMAT = 1


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 120
    batch_size: int = 24
    lr: float = 3e-3
    weight_decay: float = 1e-2
    warmup_steps: int = 150
    curriculum_epochs: int = 2  # epochs presented shortest-utterance-first
    min_lr_ratio: float = 0.05
    grad_clip: float = 5.0
    coarse_weight: float = 0.3  # weight of the articulatory-class CTC term
    inter_weight: float = 0.3  # weight of the intermediate-layer CTC term
    adapter_l2: float = 1e-2
    patience: int = 25
    seed: int = 42
    device: str = "auto"
    threads: int = 0  # intra-op CPU threads (0 = pick a small default)
    shuffle_labels: bool = False  # control: train on permuted labels (must not learn)


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)


def _say(message: str) -> None:
    print(message, flush=True)


def choose_device(name: str = "auto") -> torch.device:
    # CTC loss has no MPS kernel, so Apple GPUs are not used; the model is small
    # enough that CPU training takes minutes.
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def configure_cpu(threads: int = 0) -> None:
    """CPU settings that matter for this (small, convolution-plus-GRU) model.

    NNPACK convolutions are an order of magnitude slower than the default path on
    some ARM builds of PyTorch, and the model is too small to gain from many
    intra-op threads, so both are pinned explicitly.
    """
    if hasattr(torch.backends, "nnpack"):
        torch.backends.nnpack.set_flags(False)
    torch.set_num_threads(threads if threads > 0 else min(2, os.cpu_count() or 1))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ------------------------------------------------------------------ checkpoints
def save_checkpoint(
    path: str | Path,
    model: JawPhonemeNet,
    data: FeatureDataset,
    sessions: Sequence[str],
    config: ExperimentConfig,
    metrics: dict[str, Any],
) -> None:
    payload = {
        "format": CHECKPOINT_FORMAT,
        "model_state": model.state_dict(),
        "model_config": to_dict(model.config),
        "n_features": model.n_features,
        "inventory": data.inventory.to_dict(),
        "sessions": list(sessions),
        "norm": {"center": data.center.tolist(), "scale": data.scale.tolist()},
        "frontend": data.meta["frontend"],
        "frontend_fingerprint": data.meta["frontend_fingerprint"],
        "experiment": to_dict(config),
        "metrics": metrics,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[JawPhonemeNet, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    inventory = Inventory.from_dict(payload["inventory"])
    config = from_dict(ModelConfig, payload["model_config"])
    model = JawPhonemeNet(config, inventory.size, n_sessions=len(payload["sessions"]), n_features=payload["n_features"])
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, payload


def warm_start(model: JawPhonemeNet, path: str | Path) -> list[str]:
    """Initialise from another run (e.g. simulator pre-training); adapters are not transferred."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = {k: v for k, v in payload["model_state"].items() if not k.startswith("adapter_")}
    own = model.state_dict()
    usable = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(usable, strict=False)
    return sorted(set(own) - set(usable) - {"adapter_delta", "adapter_bias"})


# ------------------------------------------------------------------- inference
@torch.no_grad()
def predict(
    model: JawPhonemeNet,
    data: FeatureDataset,
    items: Sequence[Item],
    session_index: dict[str, int] | None = None,
    *,
    batch_size: int = 64,
    device: torch.device | str = "cpu",
) -> list[np.ndarray]:
    """Phoneme log-posteriors ``[T', V]`` for each item, in order.

    Sessions the model has no adapter for use the shared adapter, exactly as a
    new session does at run time.
    """
    model.eval()
    order = np.argsort([i.n_frames for i in items], kind="stable")
    results: list[np.ndarray | None] = [None] * len(items)
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        feats = [torch.from_numpy(data.normalize(data.load(items[i]))) for i in indices]
        lengths = torch.tensor([len(f) for f in feats])
        x = torch.zeros(len(feats), int(lengths.max()), feats[0].shape[1])
        for row, f in enumerate(feats):
            x[row, : len(f)] = f
        session = torch.tensor([(session_index or {}).get(items[i].session_key, 0) for i in indices])
        out = model(x.to(device), lengths.to(device), session.to(device))
        log_probs = torch.log_softmax(out["logits"].float(), dim=-1).cpu().numpy()
        for row, i in enumerate(indices):
            results[i] = log_probs[row, : int(out["lengths"][row])]
    return results  # type: ignore[return-value]


def score_predictions(log_probs: Sequence[np.ndarray], items: Sequence[Item], inventory: Inventory) -> dict[str, float]:
    """Greedy phoneme and articulatory-class error rates."""
    per, cer = ErrorRate(), ErrorRate()
    for lp, item in zip(log_probs, items):
        reference = inventory.encode(item.phonemes)
        per.add(reference, greedy_decode(lp))
        cer.add(inventory.to_class_ids(reference), greedy_decode(marginalize(lp, inventory)))
    return {"per": per.rate, "class_er": cer.rate, "utterances": float(per.sentences)}


# -------------------------------------------------------------------- training
def train(
    dataset_dir: str | Path,
    out_dir: str | Path,
    config: ExperimentConfig | None = None,
    *,
    init: str | Path | None = None,
    report: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    config = config or ExperimentConfig()
    report = report or _say
    cfg = config.train
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    configure_cpu(cfg.threads)
    device = choose_device(cfg.device)

    data = FeatureDataset(dataset_dir)
    inventory = data.inventory
    train_items, val_items, test_items = data.split("train"), data.split("val"), data.split("test")
    if not train_items:
        raise RuntimeError("the dataset has no training utterances")
    if not val_items:
        report("no validation split: selecting on a 10% subset of the training utterances")
        val_items = train_items[::10]
    sessions = sorted({i.session_key for i in train_items})
    session_index = {key: n + 1 for n, key in enumerate(sessions)}

    permutation = None
    if cfg.shuffle_labels:
        permutation = np.random.default_rng(cfg.seed).permutation(len(train_items)).tolist()
        report("CONTROL RUN: training labels are shuffled; error rates should stay near 100%")
    train_set = UtteranceDataset(
        data, train_items, session_index, augment_config=config.augment, seed=cfg.seed, label_permutation=permutation
    )
    sampler = BucketSampler(
        train_set.lengths, cfg.batch_size, shuffle=True, seed=cfg.seed, curriculum_epochs=cfg.curriculum_epochs
    )
    loader = DataLoader(train_set, batch_sampler=sampler, collate_fn=collate, num_workers=0)

    model = JawPhonemeNet(config.model, inventory.size, n_sessions=len(sessions), n_features=len(data.center)).to(
        device
    )
    if init is not None:
        missing = warm_start(model, init)
        report(f"warm start from {init}" + (f" (new parameters: {len(missing)})" if missing else ""))
    criterion = MultiResolutionCTC(inventory, cfg.coarse_weight, cfg.inter_weight).to(device)
    decay = [p for n, p in model.named_parameters() if p.ndim > 1 and not n.startswith("adapter_")]
    no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1 and not n.startswith("adapter_")]
    adapters = [p for n, p in model.named_parameters() if n.startswith("adapter_")]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
            {"params": adapters, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
    )
    total_steps = max(cfg.epochs * len(loader), 1)

    def schedule(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(total_steps - cfg.warmup_steps, 1)
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    save_yaml(out_dir / "config.yaml", config)
    report(
        f"model: {model.parameter_count():,} parameters | train {len(train_items)} / val {len(val_items)} / "
        f"test {len(test_items)} utterances | {len(sessions)} training sessions | device {device}"
    )

    history: list[dict[str, float]] = []
    best = {"per": math.inf, "epoch": 0}
    stale = 0
    started = time.time()
    checkpoint = out_dir / "checkpoint.pt"
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, batches = 0.0, 0
        for batch in loader:
            out = model(batch["x"].to(device), batch["lengths"].to(device), batch["session"].to(device))
            loss, _ = criterion(
                out["logits"],
                out["lengths"],
                batch["y"].to(device),
                batch["target_lengths"].to(device),
                out["inter_logits"],
            )
            if cfg.adapter_l2 > 0:
                loss = loss + cfg.adapter_l2 * model.adapter_penalty()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach())
            batches += 1

        scores = score_predictions(predict(model, data, val_items, session_index, device=device), val_items, inventory)
        row = {
            "epoch": epoch,
            "train_loss": running / max(batches, 1),
            "val_per": scores["per"],
            "val_class_er": scores["class_er"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }
        history.append(row)
        improved = scores["per"] < best["per"] - 1e-4
        if improved:
            best = {"per": scores["per"], "class_er": scores["class_er"], "epoch": epoch}
            save_checkpoint(checkpoint, model, data, sessions, config, {"val": scores, "epoch": epoch})
            stale = 0
        else:
            stale += 1
        report(
            f"epoch {epoch:3d}  loss {row['train_loss']:.3f}  val PER {scores['per']:.3f}  "
            f"class ER {scores['class_er']:.3f}  lr {row['lr']:.1e}" + ("  *" if improved else "")
        )
        if stale >= cfg.patience:
            report(f"early stop: no improvement for {cfg.patience} epochs")
            break

    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    model, payload = load_checkpoint(checkpoint, device)
    metrics: dict[str, Any] = {
        "best_epoch": best["epoch"],
        "parameters": model.parameter_count(),
        "train_seconds": round(time.time() - started, 1),
        "val": score_predictions(predict(model, data, val_items, session_index, device=device), val_items, inventory),
        "split_notes": data.meta.get("split_notes", []),
        "shuffled_labels": cfg.shuffle_labels,
    }
    if test_items:
        metrics["test"] = score_predictions(
            predict(model, data, test_items, session_index, device=device), test_items, inventory
        )
    payload["metrics"] = metrics
    torch.save(payload, checkpoint)
    save_json(out_dir / "metrics.json", metrics)
    summary = (
        f"best epoch {best['epoch']}: val PER {metrics['val']['per']:.3f}, class ER {metrics['val']['class_er']:.3f}"
    )
    if test_items:
        summary += f" | test PER {metrics['test']['per']:.3f}, class ER {metrics['test']['class_er']:.3f}"
    report(summary)
    return metrics
