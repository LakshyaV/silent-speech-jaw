"""Streaming feature front-end: raw dual-IMU frames -> articulatory features.

This is the single implementation used for dataset building *and* live
inference. It is strictly causal and stateful, so a session processed offline in
one call and the same samples arriving live in 20 ms chunks yield bit-identical
features. A model is only ever fed features whose front-end fingerprint matches
the one it was trained with.

Signal path (per session, continuous)::

    raw frames -> per-sensor timestamp resampling onto a 200 Hz grid
               -> bias removal and rotation into the canonical anatomical frame
               -> anti-alias low-pass, decimate to 100 Hz   (kinematics)
               -> high-pass |accel| envelope                (bursts, clench)
               -> rest tracking: gravity baselines, residual gyro bias, angle re-zero
               -> 23 feature channels in four stream groups

Stream groups (channel order is part of the contract)::

    jaw [0:6]    jaw angular rate (3), jaw accel minus rest gravity (3)
    ref [6:12]   same for the reference IMU, expressed in the jaw's frame
    rel [12:21]  jaw-minus-reference rate (3), accel (3), integrated angle (3)
    env [21:23]  log high-frequency vibration envelope of jaw and reference

The ``rel`` group is the jaw's motion relative to the skull with head and body
movement cancelled; its first channel is opening velocity about the hinge axis.
The raw ``jaw`` and ``ref`` groups are kept because the reference IMU, sitting
behind the ear, also carries a little articulation of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy import signal

from ..config import fingerprint
from ..io.protocol import (
    FLAG_FIFO_OVERRUN,
    MODE_FIFO,
    SENSOR_JAW,
    SENSOR_REF,
    DeviceInfo,
    split_sensors,
    to_physical,
)
from ..io.session import Session
from .calibration import Calibration
from .resample import DualResampler, TimestampSmoother

FEATURE_GROUPS: dict[str, tuple[int, int]] = {"jaw": (0, 6), "ref": (6, 12), "rel": (12, 21), "env": (21, 23)}
FEATURE_NAMES: tuple[str, ...] = (
    "jaw_wx", "jaw_wy", "jaw_wz", "jaw_ax", "jaw_ay", "jaw_az",
    "ref_wx", "ref_wy", "ref_wz", "ref_ax", "ref_ay", "ref_az",
    "rel_wx", "rel_wy", "rel_wz", "rel_ax", "rel_ay", "rel_az",
    "rel_thx", "rel_thy", "rel_thz",
    "env_jaw", "env_ref",
)  # fmt: skip
N_FEATURES = len(FEATURE_NAMES)
#: Channels holding 3-vectors (start index of each triplet), used by augmentation.
VECTOR_TRIPLETS: tuple[int, ...] = (0, 3, 6, 9, 12, 15, 18)
GYRO_TRIPLETS: tuple[int, ...] = (0, 6, 12)
FRONTEND_VERSION = 1


@dataclass(frozen=True)
class FrontendConfig:
    fs: float = 100.0  # feature rate (Hz)
    oversample: int = 2  # internal rate = fs * oversample
    lowpass_hz: float = 30.0  # kinematic band edge
    hf_highpass_hz: float = 45.0  # vibration band lower edge
    hf_smooth_hz: float = 12.0
    hf_floor_g: float = 0.002  # envelope level mapped to log(2)
    max_gap_s: float = 0.1
    smooth_fifo_timestamps: bool = True

    # Fixed scales that bring every channel to order one.
    gyro_scale_dps: float = 50.0
    accel_scale_g: float = 0.1
    angle_scale_deg: float = 5.0

    # Rest tracking.
    noise_gyro_dps: float = 0.6  # floor of the relative-rate noise scale
    noise_accel_g: float = 0.006  # floor of the relative-accel noise scale
    activity_tau_s: float = 0.04
    still_z: float = 1.6
    tilt_z: float = 3.0
    rest_hold_s: float = 0.4
    long_still_s: float = 3.0
    baseline_tau_s: float = 1.0
    bias_tau_s: float = 4.0
    bias_gate_dps: float = 2.0
    angle_rest_tau_s: float = 0.3
    warmup_s: float = 1.0

    @property
    def fs_internal(self) -> float:
        return self.fs * self.oversample

    @property
    def fingerprint(self) -> str:
        return fingerprint({"version": FRONTEND_VERSION, "features": FEATURE_NAMES, "config": self})


@dataclass
class FrontendOutput:
    """Features for consecutive frames ``k0, k0 + 1, ...`` (frame time = ``k / fs``)."""

    k0: int
    feats: np.ndarray  # [n, N_FEATURES] float32
    activity: np.ndarray  # [n] float32, robust z-score of relative jaw motion
    rest: np.ndarray  # [n] bool, jaw at rest (baselines adapting)
    valid: np.ndarray  # [n] bool, False where samples were bridged across a gap

    def __len__(self) -> int:
        return len(self.feats)

    @staticmethod
    def empty(k0: int = 0) -> FrontendOutput:
        return FrontendOutput(
            k0, np.zeros((0, N_FEATURES), np.float32), np.zeros(0, np.float32), np.zeros(0, bool), np.zeros(0, bool)
        )

    @staticmethod
    def concatenate(parts: list[FrontendOutput]) -> FrontendOutput:
        parts = [p for p in parts if len(p)]
        if not parts:
            return FrontendOutput.empty()
        return FrontendOutput(
            parts[0].k0,
            np.concatenate([p.feats for p in parts]),
            np.concatenate([p.activity for p in parts]),
            np.concatenate([p.rest for p in parts]),
            np.concatenate([p.valid for p in parts]),
        )


class _Sos:
    """Stateful second-order-section filter over ``[n, C]`` chunks."""

    def __init__(self, sos: np.ndarray):
        self.sos = sos
        self._zi: np.ndarray | None = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return x
        if self._zi is None:
            # Start in steady state for the first sample: no switch-on transient,
            # which matters for accelerometers sitting on a 1 g offset.
            self._zi = signal.sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
        y, self._zi = signal.sosfilt(self.sos, x, axis=0, zi=self._zi)
        return y


class Frontend:
    """Causal, chunk-invariant feature extractor for one continuous session."""

    def __init__(self, config: FrontendConfig | None = None, calibration: Calibration | None = None):
        self.config = config or FrontendConfig()
        self.calibration = calibration or Calibration.identity()
        cfg, cal = self.config, self.calibration
        fsi = cfg.fs_internal

        self._resampler = DualResampler(fsi, cfg.max_gap_s)
        self._smoothers: dict[int, TimestampSmoother] = {}
        self._lowpass = _Sos(signal.butter(4, cfg.lowpass_hz, "lowpass", fs=fsi, output="sos"))
        hf_edge = min(cfg.hf_highpass_hz, 0.45 * fsi)
        self._highpass = _Sos(signal.butter(2, hf_edge, "highpass", fs=fsi, output="sos"))
        self._envelope = _Sos(signal.butter(2, cfg.hf_smooth_hz, "lowpass", fs=fsi, output="sos"))

        self._R_jaw = cal.R_jaw
        self._R_ref = cal.R_ref
        self._noise_w = max(cfg.noise_gyro_dps, cal.noise_gyro_dps)
        self._noise_a = max(cfg.noise_accel_g, cal.noise_accel_g)

        dt = 1.0 / cfg.fs
        self._dt = dt
        self._k_activity = 1.0 - np.exp(-dt / cfg.activity_tau_s)
        self._k_base = 1.0 - np.exp(-dt / cfg.baseline_tau_s)
        self._k_bias = 1.0 - np.exp(-dt / cfg.bias_tau_s)
        self._k_warm = 1.0 - np.exp(-dt / 0.15)
        self._angle_leak = float(np.exp(-dt / cfg.angle_rest_tau_s))
        self._hold = max(1, round(cfg.rest_hold_s * cfg.fs))
        self._long_hold = max(self._hold, round(cfg.long_still_s * cfg.fs))
        self._warmup = round(cfg.warmup_s * cfg.fs)

        self._invalid_tail = np.zeros(cfg.oversample - 1, dtype=np.int32)

        # Tracker state.
        self._frames = 0
        self._base_j: np.ndarray | None = None
        self._base_r: np.ndarray | None = None
        self._bias_j = np.zeros(3)
        self._bias_r = np.zeros(3)
        self._theta = np.zeros(3)
        self._z = 0.0
        self._still_run = 0

    # ----------------------------------------------------------------- input
    def push(self, raw: np.ndarray, info: DeviceInfo) -> FrontendOutput:
        """Consume raw frames (both sensors, any chunking) and return new feature frames."""
        if len(raw) == 0:
            return FrontendOutput.empty()
        parts = split_sensors(raw)
        for sensor in (SENSOR_JAW, SENSOR_REF):
            part = parts[sensor]
            if len(part) == 0:
                continue
            t, accel, gyro = to_physical(part, info)
            if info.mode == MODE_FIFO and self.config.smooth_fifo_timestamps:
                smoother = self._smoothers.get(sensor)
                if smoother is None:
                    smoother = self._smoothers[sensor] = TimestampSmoother(1.0 / info.odr_hz)
                # Samples drained after a FIFO overrun follow a hole the sequence numbers do not
                # show; the clock is re-anchored there so the resampler sees the gap.
                t = smoother(part["seq"], t, (part["flags"] & FLAG_FIFO_OVERRUN) != 0)
            self._resampler.push(sensor, t, accel, gyro)
        return self._process(*self._resampler.pop())

    def backlog_s(self) -> float:
        return self._resampler.backlog_s()

    # ------------------------------------------------------------ processing
    def _process(self, k0: int, rows: np.ndarray, valid: np.ndarray) -> FrontendOutput:
        cfg, cal = self.config, self.calibration
        if len(rows) == 0:
            return FrontendOutput.empty()

        jaw_a = rows[:, 0:3] @ self._R_jaw.T
        jaw_w = (rows[:, 3:6] - cal.bias_jaw) @ self._R_jaw.T
        ref_a = rows[:, 6:9] @ self._R_ref.T
        ref_w = (rows[:, 9:12] - cal.bias_ref) @ self._R_ref.T

        high = self._highpass(np.concatenate((jaw_a, ref_a), axis=1))
        magnitude = np.stack((np.linalg.norm(high[:, 0:3], axis=1), np.linalg.norm(high[:, 3:6], axis=1)), axis=1)
        envelope = self._envelope(magnitude)
        smooth = self._lowpass(np.concatenate((jaw_w, jaw_a, ref_w, ref_a), axis=1))

        # A frame is valid only if every internal sample it covers was valid. The
        # tail of the previous chunk is carried so chunking cannot change the result.
        padded = np.concatenate((self._invalid_tail, (~valid).astype(np.int32)))
        bad = np.convolve(padded, np.ones(cfg.oversample, dtype=np.int32), mode="valid")
        self._invalid_tail = padded[len(padded) - (cfg.oversample - 1) :]

        # Decimate on absolute grid indices so chunk boundaries cannot shift the phase.
        index = k0 + np.arange(len(rows))
        pick = index % cfg.oversample == 0
        if not pick.any():
            return FrontendOutput.empty()
        frame0 = int(index[pick][0] // cfg.oversample)
        return self._track(frame0, smooth[pick], envelope[pick], bad[pick] == 0)

    def _track(self, frame0: int, y: np.ndarray, envelope: np.ndarray, valid: np.ndarray) -> FrontendOutput:
        cfg = self.config
        n = len(y)
        feats = np.empty((n, N_FEATURES), dtype=np.float32)
        activity = np.empty(n, dtype=np.float32)
        rest_flags = np.empty(n, dtype=bool)
        gs, as_, ths = 1.0 / cfg.gyro_scale_dps, 1.0 / cfg.accel_scale_g, 1.0 / cfg.angle_scale_deg
        env = np.log1p(np.maximum(envelope, 0.0) / cfg.hf_floor_g).astype(np.float32)

        for i in range(n):
            wj = y[i, 0:3] - self._bias_j
            aj = y[i, 3:6]
            wr = y[i, 6:9] - self._bias_r
            ar = y[i, 9:12]
            if self._base_j is None:
                self._base_j, self._base_r = aj.copy(), ar.copy()
            w_rel = wj - wr
            z = float(np.sqrt(np.mean((w_rel / self._noise_w) ** 2)))
            self._z += self._k_activity * (z - self._z)

            aj0 = aj - self._base_j
            ar0 = ar - self._base_r
            a_rel = aj0 - ar0
            tilt = float(np.linalg.norm(a_rel)) / self._noise_a

            self._still_run = self._still_run + 1 if self._z < cfg.still_z else 0
            warm = self._frames < self._warmup
            rest = warm or (self._still_run >= self._hold and tilt < cfg.tilt_z) or self._still_run >= self._long_hold
            if rest:
                k = self._k_warm if warm else self._k_base
                self._base_j += k * (aj - self._base_j)
                self._base_r += k * (ar - self._base_r)
                # Only a truly motionless head reveals sensor bias (not a slow head turn).
                if max(float(np.abs(wj).max()), float(np.abs(wr).max())) < cfg.bias_gate_dps:
                    kb = self._k_warm if warm else self._k_bias
                    self._bias_j += kb * wj
                    self._bias_r += kb * wr
                self._theta *= self._angle_leak
            self._theta += w_rel * self._dt

            row = feats[i]
            row[0:3] = wj * gs
            row[3:6] = aj0 * as_
            row[6:9] = wr * gs
            row[9:12] = ar0 * as_
            row[12:15] = w_rel * gs
            row[15:18] = a_rel * as_
            row[18:21] = self._theta * ths
            row[21:23] = env[i]
            activity[i] = self._z
            rest_flags[i] = rest
            self._frames += 1
        return FrontendOutput(frame0, feats, activity, rest_flags, valid.astype(bool))


def process_session(
    session: Session,
    config: FrontendConfig | None = None,
    calibration: Calibration | None = None,
    *,
    chunk_s: float = 20.0,
) -> FrontendOutput:
    """Run the front-end over a whole stored session (same code path as live)."""
    frontend = Frontend(config, calibration)
    raw = session.raw()
    if len(raw) == 0:
        return FrontendOutput.empty()
    raw = raw[np.argsort(raw["t_us"], kind="stable")]
    step = int(chunk_s * 1e6)
    edges = np.arange(int(raw["t_us"][0]), int(raw["t_us"][-1]) + step, step)
    bounds = np.searchsorted(raw["t_us"], edges)
    parts = [frontend.push(raw[a:b], session.info) for a, b in pairwise(bounds) if b > a]
    parts.append(frontend.push(raw[bounds[-1] :], session.info))
    return FrontendOutput.concatenate(parts)


def frames_for_interval(output: FrontendOutput, t_start_us: int, t_end_us: int, fs: float) -> tuple[int, int]:
    """Frame slice ``[a, b)`` of ``output`` covering a device-time interval."""
    a = int(np.floor(t_start_us * 1e-6 * fs)) - output.k0
    b = int(np.ceil(t_end_us * 1e-6 * fs)) - output.k0
    return max(a, 0), min(max(b, 0), len(output))
