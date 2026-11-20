"""Export a trained checkpoint to a deployable ONNX bundle."""

from __future__ import annotations

import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import save_json
from ..runtime.bundle import BUNDLE_FORMAT
from .encoder import ExportWrapper

log = logging.getLogger(__name__)


def _onnx_export(wrapper: torch.nn.Module, path: Path, n_features: int, opset: int) -> None:
    example = torch.randn(1, 160, n_features)
    kwargs: dict[str, Any] = {
        "input_names": ["features"],
        "output_names": ["log_probs"],
        "dynamic_axes": {"features": {1: "frames"}, "log_probs": {1: "steps"}},
        "opset_version": opset,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:  # the TorchScript exporter handles GRUs and dynamic lengths reliably
            torch.onnx.export(wrapper, (example,), str(path), dynamo=False, **kwargs)
        except TypeError:  # torch < 2.5 has no ``dynamo`` argument
            torch.onnx.export(wrapper, (example,), str(path), **kwargs)


def export_bundle(
    checkpoint: str | Path,
    out_dir: str | Path,
    *,
    session: str | None = None,
    opset: int = 17,
    tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Write ``model.onnx`` + ``bundle.json`` and verify the export numerically.

    ``session`` bakes that session's input adapter into the graph (a model adapted
    to the current mounting); by default the shared adapter is exported, which is
    what a never-seen session uses.
    """
    import onnxruntime as ort

    from ..train.trainer import load_checkpoint

    model, payload = load_checkpoint(checkpoint)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session_id = 0
    if session is not None:
        if session not in payload["sessions"]:
            raise ValueError(f"unknown session {session!r}; the checkpoint has {payload['sessions']}")
        session_id = payload["sessions"].index(session) + 1
    wrapper = ExportWrapper(model, session_id).eval()
    path = out_dir / "model.onnx"
    _onnx_export(wrapper, path, payload["n_features"], opset)

    # Numerical parity on lengths the exporter never saw.
    runner = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    worst = 0.0
    generator = torch.Generator().manual_seed(0)
    for frames in (37, 160, 501):
        x = torch.randn(1, frames, payload["n_features"], generator=generator)
        with torch.no_grad():
            reference = wrapper(x).numpy()
        produced = runner.run(["log_probs"], {"features": x.numpy()})[0]
        if produced.shape != reference.shape:
            raise RuntimeError(f"exported model returns shape {produced.shape}, expected {reference.shape}")
        worst = max(worst, float(np.abs(produced - reference).max()))
    if worst > tolerance:
        raise RuntimeError(f"exported model deviates from the checkpoint by {worst:.2e} (tolerance {tolerance:.0e})")

    model_meta: dict[str, Any] = {
        "file": path.name,
        "parameters": model.parameter_count(),
        "subsample": model.config.subsample,
        "opset": opset,
        "max_export_error": worst,
        "config": payload["model_config"],
    }
    model_meta["bytes"] = path.stat().st_size

    meta = {
        "format": BUNDLE_FORMAT,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model_meta,
        "inventory": payload["inventory"],
        "norm": payload["norm"],
        "frontend": payload["frontend"],
        "frontend_fingerprint": payload["frontend_fingerprint"],
        "sessions": payload["sessions"],
        "baked_session": session,
        "metrics": payload.get("metrics", {}),
    }
    save_json(out_dir / "bundle.json", meta)
    return meta
