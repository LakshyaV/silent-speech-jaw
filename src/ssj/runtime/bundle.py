"""Deployable model bundles.

A bundle is a directory that contains everything needed to turn front-end
features into phoneme posteriors, and nothing that ties it to the training
environment::

    bundle.json     inventory, normalisation, front-end config + fingerprint, metrics
    model.onnx      the recogniser

The model is stored as ONNX rather than as a pickled Python object, so a bundle
trained on a laptop loads unchanged on the board regardless of library
versions. The front-end fingerprint is checked on load: a model is never run on
features computed differently from the ones it was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import from_dict, load_json
from ..dsp.frontend import FrontendConfig
from ..phonology import Inventory

BUNDLE_FORMAT = 1


class BundleError(RuntimeError):
    pass


@dataclass
class Bundle:
    directory: Path
    meta: dict[str, Any]
    inventory: Inventory
    frontend_config: FrontendConfig
    center: np.ndarray
    scale: np.ndarray
    _session: Any = None

    @classmethod
    def load(cls, directory: str | Path, *, threads: int = 1) -> Bundle:
        directory = Path(directory)
        manifest = directory / "bundle.json"
        if not manifest.exists():
            raise BundleError(f"{directory} is not a bundle (no bundle.json); create one with `ssj export`")
        meta = load_json(manifest)
        if meta.get("format") != BUNDLE_FORMAT:
            raise BundleError(f"unsupported bundle format {meta.get('format')!r}")
        frontend = from_dict(FrontendConfig, meta["frontend"])
        if frontend.fingerprint != meta["frontend_fingerprint"]:
            raise BundleError(
                "front-end mismatch: this bundle was trained with a different feature extractor "
                f"({meta['frontend_fingerprint']}) than this version of the software produces "
                f"({frontend.fingerprint}). Re-export the model with the current version."
            )
        name = meta["model"]["file"]
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise BundleError("onnxruntime is required: pip install 'silent-speech-jaw[runtime]'") from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(directory / name), sess_options=options, providers=["CPUExecutionProvider"])
        return cls(
            directory=directory,
            meta=meta,
            inventory=Inventory.from_dict(meta["inventory"]),
            frontend_config=frontend,
            center=np.asarray(meta["norm"]["center"], dtype=np.float32),
            scale=np.asarray(meta["norm"]["scale"], dtype=np.float32),
            _session=session,
        )

    @property
    def subsample(self) -> int:
        return int(self.meta["model"].get("subsample", 2))

    @property
    def frame_rate(self) -> float:
        """Output frames per second."""
        return self.frontend_config.fs / self.subsample

    def log_probs(self, feats: np.ndarray) -> np.ndarray:
        """Front-end features ``[T, C]`` -> phoneme log-posteriors ``[T', V]``."""
        if len(feats) == 0:
            return np.zeros((0, self.inventory.size), dtype=np.float32)
        x = ((np.asarray(feats, dtype=np.float32) - self.center) / self.scale)[None]
        return self._session.run(["log_probs"], {"features": x})[0][0]

    def describe(self) -> str:
        model = self.meta["model"]
        metrics = self.meta.get("metrics", {})
        test = metrics.get("test") or metrics.get("val") or {}
        parts = [
            f"{model.get('parameters', 0):,} parameters",
            f"{self.inventory.size - 1} phonemes / {self.inventory.n_classes - 1} classes",
            f"front-end {self.meta['frontend_fingerprint']}",
        ]
        if "per" in test:
            parts.append(f"PER {test['per']:.3f}, class ER {test['class_er']:.3f}")
        if self.meta.get("baked_session"):
            parts.append(f"adapted to {self.meta['baked_session']}")
        return " | ".join(parts)
