"""Live decoding loop: sample source -> front-end -> segmenter -> pipeline.

The loop is the fixed-command state machine generalised to continuous speech:
learn a baseline, wait for articulation, capture until the jaw has been quiet
long enough, decode, cool down. The streaming front-end and segmenter are the
very ones used to build the training set, so a live segment looks like a
training utterance.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..dsp.activity import Segment, Segmenter
from ..dsp.calibration import Calibration, calibrate, paired_stream
from ..dsp.frontend import Frontend, FrontendOutput
from ..io.sources import SampleSource
from ..llm.fusion import DecodeResult
from ..phonology import CONTROL_WORDS
from .pipeline import Pipeline


@dataclass
class LiveEvent:
    """One decoded utterance."""

    result: DecodeResult
    start_s: float  # device time of the segment start
    duration_s: float
    latency_s: float  # wall time from segment end to result
    control: tuple[str, ...] = ()  # control gestures found in the utterance


class FeatureRing:
    """Keeps the most recent feature frames addressable by absolute frame index."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._frames: deque[np.ndarray] = deque()
        self._first = 0  # absolute index of the first stored frame
        self._count = 0

    def append(self, output: FrontendOutput) -> None:
        if len(output) == 0:
            return
        if self._count == 0:
            self._first = output.k0
        self._frames.append(output.feats)
        self._count += len(output)
        while self._count - len(self._frames[0]) >= self.capacity:
            dropped = self._frames.popleft()
            self._first += len(dropped)
            self._count -= len(dropped)

    @property
    def end(self) -> int:
        return self._first + self._count

    def slice(self, start: int, end: int) -> np.ndarray:
        if self._count == 0:
            return np.zeros((0, 0), dtype=np.float32)
        frames = np.concatenate(list(self._frames))
        a = max(start - self._first, 0)
        b = max(min(end - self._first, len(frames)), a)
        return frames[a:b]


def run_calibration(
    source: SampleSource,
    fs_internal: float,
    *,
    say: Callable[[str], None] = print,
    on_rows: Callable[[np.ndarray], None] | None = None,
    rest_s: float = 3.0,
    head_s: float = 6.0,
    jaw_s: float = 6.0,
) -> Calibration:
    """Interactive three-phase calibration against a live source."""
    collected: list[np.ndarray] = []

    def pump(seconds: float) -> tuple[int, int]:
        while not collected:
            rows = source.read()
            if len(rows):
                collected.append(rows)
                if on_rows:
                    on_rows(rows)
        start = source.device_now_us()
        end = start + int(seconds * 1e6)
        while source.device_now_us() < end and not source.exhausted:
            rows = source.read()
            if len(rows):
                collected.append(rows)
                if on_rows:
                    on_rows(rows)
        return start, end

    phases = (
        ("rest", rest_s, "calibration 1/3  REST - mouth closed, head still"),
        ("head", head_s, "calibration 2/3  HEAD - jaw relaxed: nod, shake, then tilt your head"),
        ("jaw", jaw_s, "calibration 3/3  JAW - head still: start closed, open and close slowly"),
    )
    segments: dict[str, tuple[int, int]] = {}
    for name, seconds, text in phases:
        say(text)
        pump(0.8)
        segments[name] = pump(seconds)
    pump(0.5)
    k0, rows = paired_stream(np.concatenate(collected), source.info, fs_internal)

    def cut(name: str) -> np.ndarray:
        a, b = segments[name]
        return rows[
            max(int(np.ceil(a * 1e-6 * fs_internal)) - k0, 0) : max(int(np.floor(b * 1e-6 * fs_internal)) - k0, 0)
        ]

    calibration = calibrate(cut("rest"), cut("head"), cut("jaw"))
    calibration.quality["source"] = "routine"
    say(f"calibration: {calibration.level}")
    return calibration


class LiveDecoder:
    """Drives a pipeline from a sample source and reports decoded utterances."""

    def __init__(
        self,
        source: SampleSource,
        pipeline: Pipeline,
        *,
        calibration: Calibration | None = None,
        on_event: Callable[[LiveEvent], None] | None = None,
        hint: str = "",
    ):
        self.source = source
        self.pipeline = pipeline
        self.calibration = calibration or Calibration.identity()
        self.on_event = on_event
        self.hint = hint
        config = pipeline.bundle.frontend_config
        self.fs = config.fs
        self.frontend = Frontend(config, self.calibration)
        self.segmenter = Segmenter(pipeline.config.segmenter, self.fs)
        self.ring = FeatureRing(int((pipeline.config.segmenter.max_duration_s + 3.0) * self.fs))
        self.history: deque[str] = deque(maxlen=pipeline.config.history)
        self.events: list[LiveEvent] = []

    def feed(self, rows: np.ndarray) -> list[LiveEvent]:
        """Process newly received raw frames; returns utterances decoded as a result."""
        output = self.frontend.push(rows, self.source.info)
        if len(output) == 0:
            return []
        self.ring.append(output)
        return [self._decode(segment) for segment in self.segmenter.push(output.k0, output.activity)]

    def flush(self) -> list[LiveEvent]:
        """Decode an utterance still in progress when the stream ends."""
        return [self._decode(segment) for segment in self.segmenter.flush(self.ring.end)]

    def _decode(self, segment: Segment) -> LiveEvent:
        closed = time.perf_counter()
        feats = self.ring.slice(segment.start, segment.end)
        result = self.pipeline.decode_features(feats, list(self.history), self.hint)
        control = tuple(w for w in result.words if w in CONTROL_WORDS)
        if result.accepted and result.text:
            self.history.append(result.text)
        event = LiveEvent(
            result, segment.start / self.fs, len(segment) / self.fs, time.perf_counter() - closed, control
        )
        self.events.append(event)
        if self.on_event:
            self.on_event(event)
        return event

    def run(self, max_seconds: float | None = None) -> list[LiveEvent]:
        """Read until the source is exhausted, interrupted or ``max_seconds`` elapse."""
        started = time.monotonic()
        idle = 0
        try:
            while not self.source.exhausted:
                if max_seconds is not None and time.monotonic() - started > max_seconds:
                    break
                rows = self.source.read()
                if len(rows) == 0:
                    idle += 1
                    if idle > 20000:
                        break
                    continue
                idle = 0
                self.feed(rows)
        except KeyboardInterrupt:
            pass
        self.flush()
        return self.events
