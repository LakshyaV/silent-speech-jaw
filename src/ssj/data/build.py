"""Turn recorded sessions into a training-ready feature dataset.

Each session is processed as one continuous stream by the same front-end the
live system runs; utterances are then cut out of that stream by their recorded
cue windows and tightened to the detected articulation. The result is::

    <out>/meta.json        front-end fingerprint, inventory, normalisation, sessions
    <out>/manifest.jsonl   one line per utterance (labels, split, feature path)
    <out>/feats/...        one float32 ``[frames, channels]`` array per utterance

Evaluation splits are made **by session**. Re-mounting the sensors is the
dominant source of variation, so a random split of utterances from one session
measures memorisation of that mount, not recognition. Held-out sessions are the
default whenever more than one session per wearer exists, and the manifest
records when a within-session split had to be used instead.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import load_json, save_json, to_dict
from ..dsp.activity import SegmenterConfig, trim_to_activity
from ..dsp.calibration import calibrate_session
from ..dsp.frontend import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    VECTOR_TRIPLETS,
    FrontendConfig,
    frames_for_interval,
    process_session,
)
from ..io.session import find_sessions
from ..phonology import Inventory, Lexicon, OOVError, normalize_text

log = logging.getLogger(__name__)
DATASET_FORMAT = 1


@dataclass(frozen=True)
class DatasetConfig:
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    trim: bool = True  # tighten cue windows to detected articulation
    min_frames: int = 15
    max_frames: int = 1600
    frames_per_label: float = 4.0  # an utterance must offer this many frames per phoneme
    test_sessions: tuple[str, ...] = ()  # explicit "subject/session" keys
    val_sessions: tuple[str, ...] = ()
    holdout: str = "last"  # "last": newest session(s) per subject are held out; "none"
    within_session_val: float = 0.1  # share used when a subject has too few sessions


@dataclass
class Item:
    """One utterance of a built dataset."""

    utt_id: str
    subject: str
    session: str
    split: str
    text: str
    phonemes: list[str]
    n_frames: int
    path: str
    kind: str = "sentence"

    @property
    def session_key(self) -> str:
        return f"{self.subject}/{self.session}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "utt_id": self.utt_id, "subject": self.subject, "session": self.session, "split": self.split,
            "text": self.text, "phonemes": self.phonemes, "n_frames": self.n_frames, "path": self.path, "kind": self.kind,
        }  # fmt: skip

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Item:
        return cls(
            **{k: data[k] for k in ("utt_id", "subject", "session", "split", "text", "phonemes", "n_frames", "path")},
            kind=data.get("kind", "sentence"),
        )


def _bucket(utt_id: str) -> float:
    """Deterministic value in [0, 1) used for reproducible within-session splits."""
    return int(hashlib.sha1(utt_id.encode()).hexdigest()[:8], 16) / 0x1_0000_0000


def assign_splits(session_keys: Iterable[str], config: DatasetConfig) -> tuple[dict[str, str], list[str]]:
    """Map each session to ``train`` / ``val`` / ``test`` / ``mixed``.

    ``mixed`` means the session is split at utterance level (only when a wearer
    has too few sessions to hold one out). Returns the mapping and notes that are
    stored in the dataset metadata.
    """
    keys = sorted(session_keys)
    splits = dict.fromkeys(keys, "train")
    notes: list[str] = []
    explicit_test, explicit_val = set(config.test_sessions), set(config.val_sessions)
    unknown = (explicit_test | explicit_val) - set(keys)
    if unknown:
        raise ValueError(f"split refers to unknown sessions: {sorted(unknown)}")
    for key in explicit_test:
        splits[key] = "test"
    for key in explicit_val:
        splits[key] = "val"
    if explicit_test or explicit_val or config.holdout == "none":
        if not explicit_val and config.holdout != "none":
            notes.append("no validation session given: validation utterances are drawn from training sessions")
        return splits, notes

    by_subject: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        by_subject[key.split("/", 1)[0]].append(key)
    for subject, sessions in by_subject.items():
        if len(sessions) >= 3:
            splits[sessions[-1]] = "test"
            splits[sessions[-2]] = "val"
        elif len(sessions) == 2:
            splits[sessions[-1]] = "test"
            splits[sessions[0]] = "mixed-val"
            notes.append(f"{subject}: two sessions - the newer is the test set, validation is drawn from the older")
        else:
            splits[sessions[0]] = "mixed"
            notes.append(
                f"{subject}: a single session - train/val/test are split within it; "
                "these numbers do not measure robustness to re-mounting"
            )
    return splits, notes


def _utterance_split(session_split: str, utt_id: str, config: DatasetConfig) -> str:
    if session_split == "mixed":
        value = _bucket(utt_id)
        if value < config.within_session_val:
            return "val"
        return "test" if value < 2 * config.within_session_val else "train"
    if session_split == "mixed-val":
        return "val" if _bucket(utt_id) < config.within_session_val else "train"
    return session_split


def normalization_stats(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel centre and scale for model input.

    Each 3-vector shares one isotropic scale. Scaling axes independently would
    blow the near-silent axes (everything off the hinge axis) up to unit
    variance, turning sensor noise into a full-strength input, and would not
    commute with the rotations the calibration and augmentation apply.
    """
    center = np.zeros(feats.shape[1], dtype=np.float32)
    scale = np.ones(feats.shape[1], dtype=np.float32)
    for start in VECTOR_TRIPLETS:
        scale[start : start + 3] = max(float(np.sqrt(np.mean(feats[:, start : start + 3] ** 2))), 1e-3)
    a, b = FEATURE_GROUPS["env"]
    center[a:b] = np.median(feats[:, a:b], axis=0)
    scale[a:b] = np.maximum(np.std(feats[:, a:b] - center[a:b], axis=0), 1e-3)
    return center, scale


def build_dataset(
    raw_root: str | Path,
    out_dir: str | Path,
    lexicon: Lexicon,
    inventory: Inventory,
    config: DatasetConfig | None = None,
) -> dict[str, Any]:
    """Process every session under ``raw_root`` into ``out_dir``; returns a summary."""
    config = config or DatasetConfig()
    out_dir = Path(out_dir)
    sessions = list(find_sessions(raw_root))
    if not sessions:
        raise FileNotFoundError(f"no sessions found under {raw_root}")
    splits, notes = assign_splits((s.key for s in sessions), config)
    fs = config.frontend.fs

    items: list[Item] = []
    session_meta: dict[str, Any] = {}
    dropped: dict[str, int] = defaultdict(int)
    for session in sessions:
        calibration = calibrate_session(session, config.frontend.fs_internal)
        output = process_session(session, config.frontend, calibration)
        folder = out_dir / "feats" / session.subject / session.name
        folder.mkdir(parents=True, exist_ok=True)
        kept = 0
        for utterance in session.utterances():
            try:
                phonemes = (
                    list(utterance.phonemes)
                    if utterance.phonemes
                    else lexicon.phonemize(normalize_text(utterance.text))
                )
            except OOVError as error:
                log.warning("%s/%s: %s", session.key, utterance.utt_id, error)
                dropped["out_of_vocabulary"] += 1
                continue
            if any(p not in inventory for p in phonemes):
                dropped["unknown_phoneme"] += 1
                continue
            a, b = frames_for_interval(output, utterance.t_start_us, utterance.t_end_us, fs)
            if b - a < config.min_frames:
                dropped["outside_recording"] += 1
                continue
            if config.trim:
                ta, tb = trim_to_activity(output.activity[a:b], config.segmenter, fs)
                a, b = a + ta, a + tb
            feats = output.feats[a:b]
            n = len(feats)
            if n < config.min_frames or n < config.frames_per_label * len(phonemes):
                dropped["too_short"] += 1
                continue
            if n > config.max_frames:
                dropped["too_long"] += 1
                continue
            if not output.valid[a:b].mean() > 0.98:
                dropped["stream_gaps"] += 1
                continue
            relative = Path("feats") / session.subject / session.name / f"{utterance.utt_id}.npy"
            np.save(out_dir / relative, feats.astype(np.float32))
            utt_id = f"{session.key}/{utterance.utt_id}"
            items.append(
                Item(
                    utt_id=utt_id,
                    subject=session.subject,
                    session=session.name,
                    split=_utterance_split(splits[session.key], utt_id, config),
                    text=" ".join(normalize_text(utterance.text)),
                    phonemes=phonemes,
                    n_frames=n,
                    path=str(relative),
                    kind=utterance.kind,
                )
            )
            kept += 1
        session_meta[session.key] = {
            "split": splits[session.key],
            "utterances": kept,
            "duration_s": round(len(output) / fs, 2),
            "calibration": calibration.to_dict(),
            "simulated": bool(session.meta.get("simulated", False)),
        }
        log.info("%s: %d utterances, calibration=%s", session.key, kept, calibration.level)

    if not items:
        raise RuntimeError(f"no usable utterances found under {raw_root} (dropped: {dict(dropped)})")
    train = [i for i in items if i.split == "train"] or items
    stacked = np.concatenate([np.load(out_dir / i.path) for i in train[:4000]], axis=0)
    center, scale = normalization_stats(stacked)

    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    counts = {split: sum(1 for i in items if i.split == split) for split in ("train", "val", "test")}
    meta = {
        "format": DATASET_FORMAT,
        "frontend": to_dict(config.frontend),
        "frontend_fingerprint": config.frontend.fingerprint,
        "feature_names": list(FEATURE_NAMES),
        "inventory": inventory.to_dict(),
        "norm": {"center": center.tolist(), "scale": scale.tolist()},
        "sessions": session_meta,
        "splits": counts,
        "split_notes": notes,
        "dropped": dict(dropped),
        "frames": int(sum(i.n_frames for i in items)),
    }
    save_json(out_dir / "meta.json", meta)
    return meta


class FeatureDataset:
    """Read access to a built dataset (no torch dependency)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.meta: dict[str, Any] = load_json(self.root / "meta.json")
        with (self.root / "manifest.jsonl").open("r", encoding="utf-8") as handle:
            self.items = [Item.from_dict(json.loads(line)) for line in handle if line.strip()]
        self.inventory = Inventory.from_dict(self.meta["inventory"])
        self.center = np.asarray(self.meta["norm"]["center"], dtype=np.float32)
        self.scale = np.asarray(self.meta["norm"]["scale"], dtype=np.float32)

    @property
    def frontend_config(self) -> FrontendConfig:
        from ..config import from_dict

        return from_dict(FrontendConfig, self.meta["frontend"])

    @property
    def sessions(self) -> list[str]:
        return sorted(self.meta["sessions"])

    def split(self, name: str) -> list[Item]:
        return [i for i in self.items if i.split == name]

    def load(self, item: Item) -> np.ndarray:
        """Raw (un-normalised) features of one utterance."""
        return np.load(self.root / item.path)

    def normalize(self, feats: np.ndarray) -> np.ndarray:
        return ((feats - self.center) / self.scale).astype(np.float32)
