"""Automatic quality checks applied to every recorded utterance.

Bad trials are cheap to catch while the wearer is still in front of the screen
and expensive to find later, so each trial is checked the moment it ends and the
wearer is told what to change. The same checks run when a dataset is built.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..dsp.activity import SegmenterConfig, segment_activity, session_health
from ..io.protocol import SENSOR_JAW, SENSOR_NAMES, SENSOR_REF, DeviceInfo, split_sensors

SATURATION_LSB = 32700


@dataclass(frozen=True)
class QCConfig:
    min_rate_fraction: float = 0.8  # effective rate must reach this share of nominal
    max_loss: float = 0.05  # tolerated share of lost frames
    max_gap_ms: float = 120.0
    min_active_s: float = 0.15  # an utterance must contain some articulation
    early_margin_s: float = 0.08  # activity this close to the cue start means "started early"
    late_margin_s: float = 0.05  # activity this close to the cue end means "cut off"
    max_head_rate_dps: float = 60.0  # reference-gyro RMS above this is flagged


@dataclass
class QCResult:
    ok: bool
    problems: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "problems": self.problems,
            "metrics": {k: round(float(v), 4) for k, v in self.metrics.items()},
        }

    def advice(self) -> str:
        return "; ".join(_ADVICE.get(p, p) for p in self.problems)


_ADVICE = {
    "low_rate": "sensor data is arriving too slowly - check the cable and firmware",
    "frame_loss": "frames are being lost - check the connection",
    "gap": "there was a gap in the data stream",
    "saturated": "a sensor saturated - re-seat it so it is not being knocked",
    "no_activity": "no jaw movement was detected - mouth the prompt a little more clearly",
    "started_early": "you started before the cue - wait for GO",
    "cut_off": "the recording ended while you were still moving - finish a little sooner",
    "head_motion": "a lot of head movement - try to keep your head steadier",
    "missing_sensor": "one of the sensors sent no data",
}


def check_stream(raw: np.ndarray, info: DeviceInfo, config: QCConfig | None = None) -> QCResult:
    """Transport-level checks on a stretch of raw frames."""
    config = config or QCConfig()
    problems: list[str] = []
    metrics: dict[str, float] = {}
    parts = split_sensors(raw)
    for sensor in (SENSOR_JAW, SENSOR_REF):
        part = parts[sensor]
        name = SENSOR_NAMES[sensor]
        if len(part) < 3:
            problems.append("missing_sensor")
            continue
        health = session_health(part["t_us"], part["seq"], info.odr_hz)
        for key in ("rate_hz", "jitter_ms", "max_gap_ms", "loss"):
            metrics[f"{name}_{key}"] = health[key]
        if health["rate_hz"] < config.min_rate_fraction * info.odr_hz:
            problems.append("low_rate")
        if health["loss"] > config.max_loss:
            problems.append("frame_loss")
        if health["max_gap_ms"] > config.max_gap_ms:
            problems.append("gap")
        if np.abs(part["accel"]).max() >= SATURATION_LSB or np.abs(part["gyro"]).max() >= SATURATION_LSB:
            problems.append("saturated")
    problems = list(dict.fromkeys(problems))
    return QCResult(not problems, problems, metrics)


def check_utterance(
    raw: np.ndarray,
    info: DeviceInfo,
    activity: np.ndarray,
    fs: float,
    *,
    config: QCConfig | None = None,
    segmenter: SegmenterConfig | None = None,
    head_rate_dps: float | None = None,
) -> QCResult:
    """Full check of one cue window: transport, articulation present, timing, head motion.

    ``activity`` is the front-end's activity track for the cue window.
    """
    config = config or QCConfig()
    result = check_stream(raw, info, config)
    problems, metrics = list(result.problems), dict(result.metrics)

    seg_cfg = segmenter or SegmenterConfig()
    quick = SegmenterConfig(
        start_z=seg_cfg.start_z, end_z=seg_cfg.end_z, start_frames=seg_cfg.start_frames,
        end_quiet_s=seg_cfg.end_quiet_s, pre_roll_s=0.0, post_roll_s=0.0,
        min_duration_s=config.min_active_s, max_duration_s=seg_cfg.max_duration_s, cooldown_s=0.0,
    )  # fmt: skip
    segments = segment_activity(activity, quick, fs)
    active = sum(len(s) for s in segments) / fs
    metrics["active_s"] = active
    metrics["window_s"] = len(activity) / fs
    metrics["peak_activity"] = float(activity.max()) if len(activity) else 0.0
    if not segments:
        problems.append("no_activity")
    else:
        first, last = min(s.start for s in segments), max(s.end for s in segments)
        metrics["onset_s"] = first / fs
        metrics["offset_s"] = last / fs
        if first <= config.early_margin_s * fs:
            problems.append("started_early")
        if last >= len(activity) - config.late_margin_s * fs:
            problems.append("cut_off")
    if head_rate_dps is not None:
        metrics["head_rate_dps"] = head_rate_dps
        if head_rate_dps > config.max_head_rate_dps:
            problems.append("head_motion")
    problems = list(dict.fromkeys(problems))
    return QCResult(not problems, problems, metrics)
