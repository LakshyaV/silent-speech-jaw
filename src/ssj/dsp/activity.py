"""Articulation activity detection and utterance segmentation.

The front-end emits, for every frame, a robust z-score of the jaw's motion
relative to the skull. This module turns that score into utterance boundaries
with a hysteresis state machine: a clear onset opens a segment, and the segment
closes once the jaw has been quiet for long enough to be a real pause rather
than the gap between two words.

The same segmenter trims recorded utterances for training and endpoints live
speech, so the model sees the same kind of segment in both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegmenterConfig:
    start_z: float = 3.0  # onset threshold
    end_z: float = 1.6  # quiet threshold (hysteresis)
    start_frames: int = 3  # consecutive frames above start_z to open a segment
    end_quiet_s: float = 0.7  # quiet time that closes a segment
    pre_roll_s: float = 0.25  # context kept before the onset
    post_roll_s: float = 0.20  # context kept after the last active frame
    min_duration_s: float = 0.25  # shorter bursts are discarded (swallows, twitches)
    max_duration_s: float = 12.0  # forced close
    cooldown_s: float = 0.2  # dead time after a segment closes


@dataclass(frozen=True)
class Segment:
    start: int  # first frame (absolute index), inclusive
    end: int  # last frame, exclusive
    forced: bool = False  # closed by max_duration rather than by silence

    def __len__(self) -> int:
        return self.end - self.start


class Segmenter:
    """Streaming hysteresis segmenter over per-frame activity scores."""

    def __init__(self, config: SegmenterConfig | None = None, fs: float = 100.0):
        self.config = config or SegmenterConfig()
        self.fs = fs
        cfg = self.config
        self._quiet_frames = max(1, round(cfg.end_quiet_s * fs))
        self._pre = round(cfg.pre_roll_s * fs)
        self._post = round(cfg.post_roll_s * fs)
        self._min = round(cfg.min_duration_s * fs)
        self._max = round(cfg.max_duration_s * fs)
        self._cooldown = round(cfg.cooldown_s * fs)
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._high_run = 0
        self._onset = 0
        self._last_active = 0
        self._blocked_until = -1

    @property
    def active(self) -> bool:
        """True while an utterance is in progress."""
        return self._active

    @property
    def onset(self) -> int:
        """Onset frame (including pre-roll) of the utterance in progress."""
        return self._onset - self._pre

    def push(self, k0: int, activity: np.ndarray) -> list[Segment]:
        """Feed scores for frames ``k0, k0 + 1, ...``; return segments that just closed."""
        cfg = self.config
        closed: list[Segment] = []
        for offset in range(len(activity)):
            k = k0 + offset
            score = float(activity[offset])
            if not self._active:
                if k < self._blocked_until:
                    self._high_run = 0
                    continue
                self._high_run = self._high_run + 1 if score >= cfg.start_z else 0
                if self._high_run >= cfg.start_frames:
                    self._active = True
                    self._onset = k - self._high_run + 1
                    self._last_active = k
                    self._high_run = 0
                continue

            if score >= cfg.end_z:
                self._last_active = k
            duration = k - self._onset + 1
            quiet = k - self._last_active
            if quiet >= self._quiet_frames or duration >= self._max:
                forced = quiet < self._quiet_frames
                if self._last_active - self._onset + 1 >= self._min:
                    end = min(self._last_active + 1 + self._post, k + 1)
                    closed.append(Segment(self._onset - self._pre, end, forced))
                self._active = False
                self._blocked_until = k + 1 + self._cooldown
        return closed

    def flush(self, k_end: int) -> list[Segment]:
        """Close a segment still open when the stream ends at frame ``k_end``."""
        if not self._active:
            return []
        self._active = False
        if self._last_active - self._onset + 1 < self._min:
            return []
        return [Segment(self._onset - self._pre, min(self._last_active + 1 + self._post, k_end), True)]


def segment_activity(
    activity: np.ndarray, config: SegmenterConfig | None = None, fs: float = 100.0, k0: int = 0
) -> list[Segment]:
    """Segment a complete score track."""
    segmenter = Segmenter(config, fs)
    segments = segmenter.push(k0, activity)
    segments.extend(segmenter.flush(k0 + len(activity)))
    return segments


def trim_to_activity(
    activity: np.ndarray,
    config: SegmenterConfig | None = None,
    fs: float = 100.0,
) -> tuple[int, int]:
    """Tightest ``[a, b)`` spanning all detected articulation in a cue window.

    Recorded utterances are cut by the prompt window, which contains reaction
    time and trailing stillness. Trimming to the detected activity (with the same
    pre/post-roll the live segmenter uses) removes that dead time; if nothing is
    detected the whole window is kept and the caller can flag the utterance.
    """
    segments = segment_activity(activity, config, fs)
    if not segments:
        return 0, len(activity)
    return max(min(s.start for s in segments), 0), min(max(s.end for s in segments), len(activity))


def session_health(t_us: np.ndarray, seq: np.ndarray, nominal_hz: float) -> dict[str, float]:
    """Stream quality metrics for one sensor: delivered rate, timestamp jitter, gaps, loss.

    The rate is the number of samples delivered per second of stream, not the
    reciprocal of a typical interval. A board that falls behind and drops whole
    batches, or one that stamps samples on its own polling grid, produces
    intervals that look nominal while a share of the signal is missing; counting
    samples over elapsed time cannot be fooled that way. ``loss`` covers frames
    that were numbered and never arrived; jitter is measured around the mean
    period, so it describes the timestamps as the host receives them.
    """
    if len(t_us) < 3:
        return {"rate_hz": 0.0, "nominal_hz": float(nominal_hz), "jitter_ms": 0.0, "max_gap_ms": 0.0, "loss": 1.0}
    t_ms = np.sort(t_us.astype(np.float64)) * 1e-3
    dt = np.diff(t_ms)
    period = float(t_ms[-1] - t_ms[0]) / (len(t_ms) - 1)
    span = float(seq.max() - seq.min()) + 1.0
    return {
        "rate_hz": 1000.0 / period if period > 0 else 0.0,
        "nominal_hz": float(nominal_hz),
        "jitter_ms": float(np.percentile(np.abs(dt - period), 95)),
        "max_gap_ms": float(dt.max()),
        "loss": float(max(0.0, 1.0 - len(seq) / span)),
    }
