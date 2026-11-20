"""On-disk recording sessions.

A session is one continuous wear of the sensors - the unit across which
mounting, and therefore the signal, can change. It is stored as a directory::

    <root>/<subject>/<session>/
        session.json    metadata: subject, device info, protocol, notes
        samples.bin     append-only raw frames (``RAW_DTYPE``), both sensors
        events.jsonl    calibration segments and utterances, in device time

Samples are stored exactly as received (raw LSBs, device clock). Every
derived quantity is recomputed from this record, so improvements to the
front-end apply retroactively to everything ever recorded.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..config import load_json, save_json
from .protocol import RAW_DTYPE, SENSOR_JAW, SENSOR_REF, DeviceInfo, split_sensors, to_physical

SESSION_FORMAT = 1
CALIBRATION_PHASES = ("rest", "head", "jaw")


@dataclass
class Utterance:
    utt_id: str
    text: str
    t_start_us: int
    t_end_us: int
    status: str = "ok"  # ok | bad | skipped
    kind: str = "sentence"  # word | phrase | sentence | gesture
    prompt_id: str | None = None
    phonemes: list[str] | None = None  # explicit label, overrides the lexicon
    qc: dict[str, Any] = field(default_factory=dict)

    def to_event(self) -> dict[str, Any]:
        event = {
            "type": "utterance",
            "utt_id": self.utt_id,
            "text": self.text,
            "t_start_us": int(self.t_start_us),
            "t_end_us": int(self.t_end_us),
            "status": self.status,
            "kind": self.kind,
        }
        if self.prompt_id is not None:
            event["prompt_id"] = self.prompt_id
        if self.phonemes is not None:
            event["phonemes"] = list(self.phonemes)
        if self.qc:
            event["qc"] = self.qc
        return event

    @classmethod
    def from_event(cls, event: dict[str, Any]) -> Utterance:
        return cls(
            utt_id=str(event["utt_id"]),
            text=str(event["text"]),
            t_start_us=int(event["t_start_us"]),
            t_end_us=int(event["t_end_us"]),
            status=str(event.get("status", "ok")),
            kind=str(event.get("kind", "sentence")),
            prompt_id=event.get("prompt_id"),
            phonemes=event.get("phonemes"),
            qc=dict(event.get("qc") or {}),
        )


class SessionWriter:
    """Incrementally writes a session; safe against interruption at any point."""

    def __init__(
        self, directory: str | Path, *, subject: str, session: str, info: DeviceInfo, meta: dict[str, Any] | None = None
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / "samples.bin").exists():
            raise FileExistsError(f"{self.directory} already contains a recording")
        self.meta: dict[str, Any] = {
            "format": SESSION_FORMAT,
            "subject": subject,
            "session": session,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "device": info.to_dict(),
            **(meta or {}),
        }
        self._samples = (self.directory / "samples.bin").open("wb")
        self._events = (self.directory / "events.jsonl").open("w", encoding="utf-8")
        self.n_samples = 0
        self._last_flush = time.monotonic()
        save_json(self.directory / "session.json", self.meta)

    def append(self, raw: np.ndarray) -> None:
        if raw.dtype != RAW_DTYPE:
            raise TypeError("samples must use RAW_DTYPE")
        if len(raw) == 0:
            return
        self._samples.write(raw.tobytes())
        self.n_samples += len(raw)
        if time.monotonic() - self._last_flush > 0.5:
            self._samples.flush()
            self._last_flush = time.monotonic()

    def event(self, event: dict[str, Any]) -> None:
        self._events.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._events.flush()
        self._samples.flush()

    def utterance(self, utterance: Utterance) -> None:
        self.event(utterance.to_event())

    def calibration(self, phase: str, t_start_us: int, t_end_us: int) -> None:
        if phase not in CALIBRATION_PHASES:
            raise ValueError(f"unknown calibration phase {phase!r}")
        self.event({"type": "calibration", "phase": phase, "t_start_us": int(t_start_us), "t_end_us": int(t_end_us)})

    def close(self, **extra_meta: Any) -> None:
        if self._samples.closed:
            return
        self._samples.close()
        self._events.close()
        self.meta.update(extra_meta)
        self.meta["n_samples"] = self.n_samples
        save_json(self.directory / "session.json", self.meta)

    def __enter__(self) -> SessionWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class Session:
    """Read-only view of a recorded (or simulated) session."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        if not (self.directory / "session.json").exists():
            raise FileNotFoundError(f"{self.directory} is not a session directory")
        self.meta: dict[str, Any] = load_json(self.directory / "session.json")
        self.info = DeviceInfo.from_dict(self.meta["device"])
        self.events: list[dict[str, Any]] = []
        events_path = self.directory / "events.jsonl"
        if events_path.exists():
            with events_path.open("r", encoding="utf-8") as handle:
                self.events = [json.loads(line) for line in handle if line.strip()]

    # ------------------------------------------------------------- identity
    @property
    def subject(self) -> str:
        return str(self.meta["subject"])

    @property
    def name(self) -> str:
        return str(self.meta["session"])

    @property
    def key(self) -> str:
        """Globally unique ``subject/session`` identifier."""
        return f"{self.subject}/{self.name}"

    # --------------------------------------------------------------- samples
    def raw(self) -> np.ndarray:
        path = self.directory / "samples.bin"
        usable = (path.stat().st_size // RAW_DTYPE.itemsize) * RAW_DTYPE.itemsize
        if usable == 0:
            return np.zeros(0, dtype=RAW_DTYPE)
        return np.fromfile(path, dtype=RAW_DTYPE, count=usable // RAW_DTYPE.itemsize)

    def streams(self) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Per-sensor ``(t_s, accel_g, gyro_dps)`` in physical units."""
        parts = split_sensors(self.raw())
        return {"jaw": to_physical(parts[SENSOR_JAW], self.info), "ref": to_physical(parts[SENSOR_REF], self.info)}

    # ---------------------------------------------------------------- events
    def utterances(self, *, include_bad: bool = False) -> list[Utterance]:
        items = [Utterance.from_event(e) for e in self.events if e.get("type") == "utterance"]
        return items if include_bad else [u for u in items if u.status == "ok"]

    def calibration_segments(self) -> dict[str, tuple[int, int]]:
        """Latest ``(t_start_us, t_end_us)`` for each recorded calibration phase."""
        segments: dict[str, tuple[int, int]] = {}
        for event in self.events:
            if event.get("type") == "calibration":
                segments[str(event["phase"])] = (int(event["t_start_us"]), int(event["t_end_us"]))
        return segments

    def duration_s(self) -> float:
        raw = self.raw()
        if len(raw) < 2:
            return 0.0
        return float(raw["t_us"].max() - raw["t_us"].min()) * 1e-6


def find_sessions(root: str | Path) -> Iterator[Session]:
    """Yield every session under ``root`` in a stable (subject, session) order."""
    root = Path(root)
    for meta in sorted(root.rglob("session.json")):
        yield Session(meta.parent)
