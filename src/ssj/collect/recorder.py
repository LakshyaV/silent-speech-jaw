"""Guided recording session: calibration routine, cued prompts, live quality checks.

The host drives the protocol; the firmware only streams. Each trial follows the
three-phase structure that proved reliable for fixed commands - stillness, a cued
articulation window, stillness - but the window length now follows the prompt's
phoneme count, and the label is the prompt's phoneme sequence rather than a
class index. Alignment inside the window is left to CTC.

All waiting is done in *device* time (the clock that stamps the samples), so the
recorded event boundaries are exact and the same code runs against a stored
session for rehearsal and tests.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ..dsp.activity import SegmenterConfig
from ..dsp.calibration import Calibration, calibrate, paired_stream
from ..dsp.frontend import FEATURE_GROUPS, Frontend, FrontendConfig
from ..io.protocol import RAW_DTYPE
from ..io.session import SessionWriter, Utterance
from ..io.sources import SampleSource
from ..phonology import CONTROL_WORDS
from .prompts import Prompt
from .qc import QCConfig, QCResult, check_stream, check_utterance


class EndOfStream(Exception):
    """The source stopped delivering samples."""


@dataclass(frozen=True)
class RecorderConfig:
    pre_rest_s: float = 1.0
    post_rest_s: float = 0.6
    base_window_s: float = 0.9
    per_phoneme_s: float = 0.17
    min_window_s: float = 1.6
    max_window_s: float = 10.0
    lead_margin_s: float = 0.3  # of the pre-rest kept inside the stored window
    tail_margin_s: float = 0.4
    calib_rest_s: float = 3.0
    calib_head_s: float = 6.0
    calib_jaw_s: float = 6.0
    auto: bool = False  # hands-free: accept clean trials, retry flagged ones
    max_retries: int = 2

    def window_s(self, n_phonemes: int) -> float:
        return float(
            np.clip(self.base_window_s + self.per_phoneme_s * n_phonemes, self.min_window_s, self.max_window_s)
        )


class ConsoleUI:
    """Minimal terminal interface; replaced by a scripted one in tests."""

    def __init__(self, ask: Callable[[str], str] | None = None, out=None):
        self._ask = ask or input
        self._out = out or sys.stdout

    def say(self, text: str = "") -> None:
        print(text, file=self._out, flush=True)

    def banner(self, text: str) -> None:
        bar = "=" * 68
        self.say(f"\n{bar}\n{text}\n{bar}")

    def cue(self, text: str) -> None:
        self.say(f"\n    >>>  {text}  <<<\n")

    def ask(self, prompt: str) -> str:
        try:
            return self._ask(prompt).strip().lower()
        except EOFError:
            return "q"


class Recorder:
    def __init__(
        self,
        source: SampleSource,
        writer: SessionWriter,
        *,
        config: RecorderConfig | None = None,
        frontend_config: FrontendConfig | None = None,
        qc_config: QCConfig | None = None,
        segmenter: SegmenterConfig | None = None,
        ui: ConsoleUI | None = None,
    ):
        self.source = source
        self.writer = writer
        self.config = config or RecorderConfig()
        self.frontend_config = frontend_config or FrontendConfig()
        self.qc_config = qc_config or QCConfig()
        self.segmenter = segmenter or SegmenterConfig()
        self.ui = ui or ConsoleUI()
        self.calibration: Calibration | None = None
        self._frontend: Frontend | None = None
        self._raw: list[np.ndarray] = []
        self._k0: int | None = None
        self._activity: list[np.ndarray] = []
        self._feats: list[np.ndarray] = []
        self._idle_reads = 0

    # ------------------------------------------------------------- plumbing
    def _now(self) -> int:
        return self.source.device_now_us()

    def _ingest(self) -> None:
        rows = self.source.read()
        if len(rows) == 0:
            self._idle_reads += 1
            if self.source.exhausted or self._idle_reads > 5000:
                raise EndOfStream
            return
        self._idle_reads = 0
        self.writer.append(rows)
        self._raw.append(rows)
        if self._frontend is not None:
            out = self._frontend.push(rows, self.source.info)
            if len(out):
                if self._k0 is None:
                    self._k0 = out.k0
                self._activity.append(out.activity)
                self._feats.append(out.feats)

    def _wait_first_sample(self) -> None:
        while not self._raw:
            self._ingest()

    def _pump_until(self, t_us: int) -> None:
        while self._now() < t_us:
            self._ingest()

    def _pump_for(self, seconds: float) -> tuple[int, int]:
        start = self._now()
        end = start + int(seconds * 1e6)
        self._pump_until(end)
        return start, end

    def _raw_between(self, t_start_us: int, t_end_us: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for chunk in reversed(self._raw):  # newest first; stop once we are before the window
            if int(chunk["t_us"].max()) < t_start_us - 500_000:
                break
            chunks.append(chunk)
        if not chunks:
            return np.zeros(0, dtype=RAW_DTYPE)
        raw = np.concatenate(chunks[::-1])
        return raw[(raw["t_us"] >= t_start_us) & (raw["t_us"] <= t_end_us)]

    def _tracks_between(self, t_start_us: int, t_end_us: int) -> tuple[np.ndarray, np.ndarray]:
        if self._k0 is None:
            return np.zeros(0, np.float32), np.zeros((0, 0), np.float32)
        fs = self.frontend_config.fs
        activity = np.concatenate(self._activity)
        feats = np.concatenate(self._feats)
        a = max(int(t_start_us * 1e-6 * fs) - self._k0, 0)
        b = min(int(t_end_us * 1e-6 * fs) - self._k0, len(activity))
        return activity[a:b], feats[a:b]

    # ----------------------------------------------------------- calibration
    def check_link(self, seconds: float = 2.0) -> QCResult:
        """Measure the stream before anything is recorded and report problems."""
        self._wait_first_sample()
        start, end = self._pump_for(seconds)
        result = check_stream(self._raw_between(start, end), self.source.info, self.qc_config)
        m = result.metrics
        self.ui.say(
            f"link: jaw {m.get('jaw_rate_hz', 0):.0f} Hz, ref {m.get('ref_rate_hz', 0):.0f} Hz "
            f"(nominal {self.source.info.odr_hz:.0f}); loss {100 * max(m.get('jaw_loss', 0), m.get('ref_loss', 0)):.1f}%"
        )
        if not result.ok:
            self.ui.say(f"  warning: {result.advice()}")
        return result

    def calibrate(self) -> Calibration:
        cfg, ui = self.config, self.ui
        self._wait_first_sample()
        ui.banner("CALIBRATION  (about 20 seconds)")
        phases = (
            ("rest", cfg.calib_rest_s, "1/3  REST - mouth closed, head still, look straight ahead"),
            ("head", cfg.calib_head_s, "2/3  HEAD - jaw relaxed: nod, then shake, then tilt your head"),
            ("jaw", cfg.calib_jaw_s, "3/3  JAW - head still: start closed, then open and close slowly"),
        )
        segments: dict[str, tuple[int, int]] = {}
        for phase, seconds, text in phases:
            ui.say(text)
            self._pump_for(0.8)  # settle between phases
            start, end = self._pump_for(seconds)
            segments[phase] = (start, end)
            self.writer.calibration(phase, start, end)
        self._pump_for(0.6)

        fs = self.frontend_config.fs_internal
        k0, rows = paired_stream(np.concatenate(self._raw), self.source.info, fs)

        def cut(phase: str) -> np.ndarray:
            a, b = segments[phase]
            return rows[max(int(np.ceil(a * 1e-6 * fs)) - k0, 0) : max(int(np.floor(b * 1e-6 * fs)) - k0, 0)]

        calibration = calibrate(cut("rest"), cut("head"), cut("jaw"))
        calibration.quality["source"] = "routine"
        self.calibration = calibration
        self._frontend = Frontend(self.frontend_config, calibration)
        q = calibration.quality
        ui.say(
            f"calibration: {calibration.level}"
            + (f", head-motion residual {100 * q['head_residual']:.0f}%" if "head_residual" in q else "")
            + (f", hinge axis explains {100 * q['hinge_explained']:.0f}%" if "hinge_explained" in q else "")
        )
        if calibration.level != "full":
            ui.say("  warning: calibration is incomplete - repeat it with larger, slower movements")
        self._pump_for(self.frontend_config.warmup_s + 0.2)  # let the trackers settle
        return calibration

    # ----------------------------------------------------------------- trial
    def record_prompt(self, prompt: Prompt, utt_id: str) -> tuple[Utterance, QCResult]:
        cfg, ui = self.config, self.ui
        shown = "CLENCH your teeth once" if prompt.text in CONTROL_WORDS else prompt.text
        ui.cue(shown)
        ui.say("  ready - keep still")
        self._pump_for(cfg.pre_rest_s)
        go = self._now()
        ui.say("  GO")
        cue_end = go + int(cfg.window_s(len(prompt.phonemes)) * 1e6)
        self._pump_until(cue_end)
        ui.say("  relax")
        self._pump_for(max(cfg.post_rest_s, cfg.tail_margin_s + 0.05))

        activity, feats = self._tracks_between(go, cue_end)
        a, _ = FEATURE_GROUPS["ref"]
        head = None
        if len(feats):
            head = float(
                np.sqrt(np.mean(np.sum((feats[:, a : a + 3] * self.frontend_config.gyro_scale_dps) ** 2, axis=1)))
            )
        qc = check_utterance(
            self._raw_between(go, cue_end), self.source.info, activity, self.frontend_config.fs,
            config=self.qc_config, segmenter=self.segmenter, head_rate_dps=head,
        )  # fmt: skip
        utterance = Utterance(
            utt_id=utt_id,
            text=prompt.text,
            t_start_us=go - int(cfg.lead_margin_s * 1e6),
            t_end_us=cue_end + int(cfg.tail_margin_s * 1e6),
            status="ok" if qc.ok else "bad",
            kind=prompt.kind,
            prompt_id=prompt.prompt_id,
            phonemes=list(prompt.phonemes) or None,
            qc=qc.to_dict(),
        )
        return utterance, qc

    def run(self, prompts: Sequence[Prompt], *, calibrate: bool = True) -> dict[str, int]:
        cfg, ui = self.config, self.ui
        counts = {"ok": 0, "bad": 0, "skipped": 0}
        try:
            self.check_link()
            if calibrate:
                self.calibrate()
            else:
                self._frontend = Frontend(self.frontend_config, Calibration.identity())
            ui.banner(f"RECORDING  {len(prompts)} prompts - mouth each one silently, once, at a natural pace")
            trial = 0
            for number, prompt in enumerate(prompts, start=1):
                attempts = 0
                while True:
                    ui.say(f"[{number}/{len(prompts)}] {prompt.kind}")
                    utterance, qc = self.record_prompt(prompt, f"u{trial:05d}")
                    trial += 1
                    attempts += 1
                    if qc.ok:
                        ui.say(f"  ok ({qc.metrics.get('active_s', 0):.2f} s of movement)")
                    else:
                        ui.say(f"  check: {qc.advice()}")
                    if cfg.auto:
                        choice = "" if qc.ok or attempts > cfg.max_retries else "r"
                    else:
                        hint = "Enter=next" if qc.ok else "Enter=redo  k=keep"
                        choice = ui.ask(f"  [{hint}  r=redo  s=skip  q=quit] ")
                        if not qc.ok and choice == "":
                            choice = "r"
                    if choice == "k":
                        utterance.status = "ok"
                    if choice in ("r", "s", "q") and utterance.status == "ok":
                        utterance.status = "bad"  # the wearer rejected a trial the checks passed
                    self.writer.utterance(utterance)
                    counts[utterance.status if utterance.status in counts else "bad"] += 1
                    if choice == "r" and attempts <= cfg.max_retries + 1:
                        continue
                    if choice == "s":
                        counts["skipped"] += 1
                    break
                if choice == "q":
                    break
        except EndOfStream:
            ui.say("\nstream ended")
        ui.banner(f"SESSION COMPLETE  ok={counts['ok']}  flagged={counts['bad']}  skipped={counts['skipped']}")
        return counts
