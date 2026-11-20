"""Jaw trajectory -> what the two IMUs would report.

Rigid-body kinematics of a sensor on the mandible and one on the skull:

* the skull moves with the head (slow sway plus scripted nods and shakes);
* the mandible additionally rotates about the hinge axis and slides forward;
* each IMU measures angular rate and specific force (acceleration minus
  gravity) in its *own* frame, rotated by an arbitrary mounting orientation;
* the reference IMU behind the ear picks up a fraction of the jaw motion;
* both have bias, noise, gain error, their own sample clock and quantisation.

Every nuisance the real pipeline has to survive is a parameter here, and the
ground truth (mount rotations, hinge axis, jaw angle) is returned alongside the
samples, so calibration and the front-end can be verified numerically.

Frames: the *skull* frame has x along the hinge axis (opening is a positive
rotation about x), z up when the head is upright, and y completing the
right-handed set. The *jaw* frame coincides with it when the mouth is closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal
from scipy.spatial.transform import Rotation

from ..io.protocol import FLAGS_OK, MODE_FIFO, MODE_POLLED, RAW_DTYPE, SENSOR_JAW, SENSOR_REF, DeviceInfo
from ..io.session import SessionWriter, Utterance
from ..phonology import Lexicon, normalize_text
from .articulation import Trajectory, open_close, rest, synthesize

G0 = 9.80665
FS_SIM = 400.0


@dataclass
class SessionParams:
    """Everything that changes when the device is taken off and put back on."""

    R_mount_jaw: np.ndarray  # jaw sensor frame <- jaw anatomical frame
    R_mount_ref: np.ndarray  # ref sensor frame <- skull frame
    lever_m: np.ndarray  # jaw sensor position relative to the hinge axis (jaw frame)
    leak: float  # fraction of jaw rotation sensed by the reference IMU
    leak_mix: np.ndarray  # axis mixing of that leak
    bias_jaw_dps: np.ndarray
    bias_ref_dps: np.ndarray
    gain_jaw: float
    gain_ref: float
    odr_jaw_hz: float  # true output data rates (each sensor has its own oscillator)
    odr_ref_hz: float
    gyro_noise_dps: float = 0.07
    accel_noise_g: float = 0.0015
    head_sway_dps: float = 1.5
    mode: int = MODE_FIFO

    def truth(self) -> dict[str, Any]:
        return {
            "R_ref_to_jaw": (self.R_mount_jaw @ self.R_mount_ref.T).tolist(),
            "R_canon": self.R_mount_jaw.T.tolist(),
            "hinge_axis_jaw_sensor": self.R_mount_jaw[:, 0].tolist(),
            "odr_jaw_hz": self.odr_jaw_hz,
            "odr_ref_hz": self.odr_ref_hz,
        }


def random_rotation(rng: np.random.Generator, max_angle_deg: float = 180.0) -> np.ndarray:
    """Random rotation; ``max_angle_deg`` < 180 limits it to a cone around identity."""
    if max_angle_deg >= 180.0:
        return Rotation.random(random_state=rng).as_matrix()
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return Rotation.from_rotvec(axis * np.radians(rng.uniform(0, max_angle_deg))).as_matrix()


def random_session_params(
    rng: np.random.Generator,
    *,
    mount_spread_deg: float = 180.0,
    nominal_odr_hz: float = 208.0,
    mode: int = MODE_FIFO,
    head_sway_dps: float | None = None,
) -> SessionParams:
    return SessionParams(
        R_mount_jaw=random_rotation(rng, mount_spread_deg),
        R_mount_ref=random_rotation(rng, mount_spread_deg),
        lever_m=np.array([rng.uniform(-0.005, 0.005), rng.uniform(0.045, 0.075), rng.uniform(-0.055, -0.03)]),
        leak=float(rng.uniform(0.04, 0.14)),
        leak_mix=np.eye(3) + rng.normal(0, 0.15, (3, 3)),
        bias_jaw_dps=rng.normal(0, 0.4, 3),
        bias_ref_dps=rng.normal(0, 0.4, 3),
        gain_jaw=float(rng.uniform(0.985, 1.015)),
        gain_ref=float(rng.uniform(0.985, 1.015)),
        odr_jaw_hz=nominal_odr_hz * float(rng.uniform(0.985, 1.015)),
        odr_ref_hz=nominal_odr_hz * float(rng.uniform(0.985, 1.015)),
        head_sway_dps=float(rng.uniform(0.8, 3.0)) if head_sway_dps is None else head_sway_dps,
        mode=mode,
    )


# ------------------------------------------------------------------- script
@dataclass
class ScriptItem:
    kind: str  # "rest" | "head" | "jaw" | "utterance"
    trajectory: Trajectory
    text: str = ""
    phonemes: list[str] = field(default_factory=list)
    utt_kind: str = "sentence"
    head_motion: bool = False


def calibration_script(rng: np.random.Generator) -> list[ScriptItem]:
    """The three-phase calibration routine, as the recorder runs it."""
    return [
        ScriptItem("rest", rest(3.0)),
        ScriptItem("head", rest(6.0), head_motion=True),
        ScriptItem("gap", rest(1.0)),
        ScriptItem("jaw", open_close(5, rng)),
        ScriptItem("gap", rest(1.5)),
    ]


def utterance_item(
    text: str,
    lexicon: Lexicon,
    rng: np.random.Generator,
    *,
    kind: str = "sentence",
    rate: float | None = None,
    tissue_gain: float = 1.0,
) -> ScriptItem:
    words = normalize_text(text)
    variants = [int(rng.integers(0, 4)) for _ in words]
    prons = lexicon.phonemize_words(words, variants)
    phonemes: list[str] = []
    boundaries: list[int] = []
    for pron in prons:
        boundaries.append(len(phonemes))
        phonemes.extend(pron)
    trajectory = synthesize(phonemes, rng, fs=FS_SIM, rate=rate, word_boundaries=boundaries, tissue_gain=tissue_gain)
    return ScriptItem("utterance", trajectory, " ".join(words), phonemes, kind)


# ------------------------------------------------------------------ rendering
def _smooth_noise(n: int, rng: np.random.Generator, cutoff_hz: float, fs: float, channels: int = 3) -> np.ndarray:
    sos = signal.butter(2, cutoff_hz, "lowpass", fs=fs, output="sos")
    x = signal.sosfilt(sos, rng.normal(0, 1, (n + 400, channels)), axis=0)[400:]
    return x / (x.std(axis=0, keepdims=True) + 1e-12)


def _head_angles(
    n: int, fs: float, rng: np.random.Generator, sway_dps: float, scripted: list[tuple[int, int]]
) -> np.ndarray:
    """Head orientation as small Euler angles (degrees) about x, y, z."""
    # Natural sway: band-limited angle noise whose rate has the requested RMS.
    sway = _smooth_noise(n, rng, 0.8, fs)
    rate = np.gradient(sway, axis=0) * fs
    angles = sway * (sway_dps / (rate.std() + 1e-12))
    t = np.arange(n) / fs
    for start, end in scripted:  # calibration: nod, shake, tilt in turn
        length = end - start
        third = length // 3
        window = np.hanning(third)
        for axis, amplitude, freq in ((0, 14.0, 0.9), (2, 18.0, 0.8), (1, 10.0, 0.7)):
            a = start + axis_order(axis) * third
            angles[a : a + third, axis] += amplitude * window * np.sin(2 * np.pi * freq * t[:third])
    return angles


def axis_order(axis: int) -> int:
    """Position of each axis in the scripted head sequence (nod x, shake z, tilt y)."""
    return {0: 0, 2: 1, 1: 2}[axis]


def _sample_times(
    duration_s: float, odr_hz: float, mode: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """True sampling instants and the timestamps the firmware would report."""
    if mode == MODE_FIFO:
        true = np.arange(0.0, duration_s, 1.0 / odr_hz) + rng.uniform(0, 1.0 / odr_hz)
        # The MCU stamps samples when it drains the FIFO: quantised to its poll period.
        poll = 0.004
        reported = np.ceil(true / poll) * poll
        return true, reported
    period = 1.0 / odr_hz
    steps = period * (1.0 + np.abs(rng.normal(0, 0.08, int(duration_s / period) + 8)))
    steps[rng.random(len(steps)) < 0.01] *= 2.0  # occasional late poll
    true = np.cumsum(steps)
    true = true[true < duration_s]
    return true, true + rng.uniform(0.0002, 0.0006, len(true))


@dataclass
class SimSession:
    raw: np.ndarray
    info: DeviceInfo
    events: list[dict[str, Any]]
    utterances: list[Utterance]
    truth: dict[str, Any]
    params: SessionParams
    # Ground-truth kinematics on the simulation grid, for tests and plots.
    fs: float = FS_SIM
    opening_deg: np.ndarray | None = None

    def write(self, directory: str | Path, *, subject: str, session: str, meta: dict[str, Any] | None = None) -> Path:
        directory = Path(directory)
        with SessionWriter(
            directory,
            subject=subject,
            session=session,
            info=self.info,
            meta={"simulated": True, "truth": self.truth, **(meta or {})},
        ) as writer:
            writer.append(self.raw)
            for event in self.events:
                writer.event(event)
        return directory


def render_session(
    script: list[ScriptItem],
    params: SessionParams,
    rng: np.random.Generator,
    *,
    info: DeviceInfo | None = None,
    gap_s: tuple[float, float] = (0.9, 1.6),
    t0_s: float = 5.0,
    utt_prefix: str = "utt",
) -> SimSession:
    """Render a scripted session into raw frames, events and ground truth."""
    fs = FS_SIM
    info = info or DeviceInfo(mode=params.mode, odr_hz=208.0, firmware=(0, 1, 0))

    # 1. Lay the script out on one timeline ---------------------------------
    pieces: list[Trajectory] = []
    spans: list[tuple[ScriptItem, int, int]] = []
    cursor = 0
    for item in script:
        if item.kind == "utterance":
            gap = rest(float(rng.uniform(*gap_s)), fs)
            pieces.append(gap)
            cursor += len(gap)
        pieces.append(item.trajectory)
        spans.append((item, cursor, cursor + len(item.trajectory)))
        cursor += len(item.trajectory)
    pieces.append(rest(1.2, fs))
    n = sum(len(p) for p in pieces)
    opening = np.concatenate([p.opening_deg for p in pieces])
    protrusion = np.concatenate([p.protrusion_mm for p in pieces]) * 1e-3
    lateral = np.concatenate([p.lateral_deg for p in pieces])
    tissue = np.concatenate([p.tissue_deg for p in pieces])
    vibration = np.concatenate([p.vibration_g for p in pieces])

    # 2. Head motion --------------------------------------------------------
    scripted = [(a, b) for item, a, b in spans if item.head_motion]
    head = _head_angles(n, fs, rng, params.head_sway_dps, scripted)
    head_rate = np.gradient(head, axis=0) * fs  # deg/s in the skull frame (small angles)
    gravity_skull = Rotation.from_euler("xyz", head, degrees=True).inv().apply([0.0, 0.0, 1.0])

    # 3. Jaw relative to skull ----------------------------------------------
    # Sensor orientation relative to the skull: hinge rotation plus soft-tissue tilt.
    pose = np.stack((opening, tissue[:, 0], tissue[:, 1] + lateral), axis=1)
    rel_rate = np.gradient(pose, axis=0) * fs
    R_open = Rotation.from_euler("xyz", pose, degrees=True)  # jaw sensor frame -> skull frame
    jaw_rate = R_open.inv().apply(head_rate) + rel_rate  # deg/s, jaw frame

    omega = np.radians(jaw_rate)
    alpha = np.gradient(omega, axis=0) * fs
    lever = params.lever_m
    linear = np.cross(alpha, lever) + np.cross(omega, np.cross(omega, lever))
    linear[:, 1] += np.gradient(np.gradient(protrusion)) * fs * fs
    jaw_force = R_open.inv().apply(gravity_skull) + linear / G0 + vibration

    ref_rate = head_rate + params.leak * (rel_rate @ params.leak_mix.T)
    ref_force = gravity_skull + 0.25 * params.leak * (linear / G0) + 0.3 * vibration

    # 4. Mounting, gain, bias, noise ----------------------------------------
    def sensor(rate, force, R_mount, gain, bias, odr, sensor_id):
        w = (rate @ R_mount.T) * gain + bias
        f = (force @ R_mount.T) * gain
        true_t, reported_t = _sample_times(n / fs, odr, params.mode, rng)
        grid = np.arange(n) / fs
        w_s = np.stack([np.interp(true_t, grid, w[:, c]) for c in range(3)], axis=1)
        f_s = np.stack([np.interp(true_t, grid, f[:, c]) for c in range(3)], axis=1)
        w_s += rng.normal(0, params.gyro_noise_dps, w_s.shape)
        f_s += rng.normal(0, params.accel_noise_g, f_s.shape)
        rows = np.zeros(len(true_t), dtype=RAW_DTYPE)
        rows["t_us"] = np.round((t0_s + reported_t) * 1e6).astype(np.int64)
        rows["seq"] = np.arange(len(true_t), dtype=np.uint32)
        rows["sensor"] = sensor_id
        rows["flags"] = FLAGS_OK
        rows["accel"] = np.clip(np.round(f_s / info.accel_g_per_lsb), -32768, 32767).astype(np.int16)
        rows["gyro"] = np.clip(np.round(w_s / info.gyro_dps_per_lsb), -32768, 32767).astype(np.int16)
        return rows

    jaw_rows = sensor(
        jaw_rate, jaw_force, params.R_mount_jaw, params.gain_jaw, params.bias_jaw_dps, params.odr_jaw_hz, SENSOR_JAW
    )
    ref_rows = sensor(
        ref_rate, ref_force, params.R_mount_ref, params.gain_ref, params.bias_ref_dps, params.odr_ref_hz, SENSOR_REF
    )
    raw = np.concatenate((jaw_rows, ref_rows))
    raw = raw[np.argsort(raw["t_us"], kind="stable")]

    # 5. Events -------------------------------------------------------------
    def to_us(index: int) -> int:
        return round((t0_s + index / fs) * 1e6)

    events: list[dict[str, Any]] = []
    utterances: list[Utterance] = []
    count = 0
    for item, a, b in spans:
        if item.kind in ("rest", "head", "jaw"):
            events.append({"type": "calibration", "phase": item.kind, "t_start_us": to_us(a), "t_end_us": to_us(b)})
        elif item.kind == "utterance":
            # The cue window includes reaction time before and stillness after.
            lead = int(rng.uniform(0.25, 0.5) * fs)
            lag = int(rng.uniform(0.3, 0.6) * fs)
            utterance = Utterance(
                utt_id=f"{utt_prefix}{count:05d}",
                text=item.text,
                t_start_us=to_us(max(a - lead, 0)),
                t_end_us=to_us(min(b + lag, n)),
                kind=item.utt_kind,
                phonemes=list(item.phonemes),
            )
            utterances.append(utterance)
            events.append(utterance.to_event())
            count += 1
    return SimSession(raw, info, events, utterances, params.truth(), params, fs, opening)


def simulate_session(
    texts: list[str],
    lexicon: Lexicon,
    seed: int = 0,
    *,
    calibrate: bool = True,
    params: SessionParams | None = None,
    mode: int = MODE_FIFO,
    mount_spread_deg: float = 180.0,
    clench_rate: float = 0.0,
    utt_prefix: str = "utt",
    tissue_gain: float = 1.0,
) -> SimSession:
    """Simulate one recording session of the given prompts.

    ``clench_rate`` is the probability that an utterance is followed by the
    ``<clench>`` control gesture, so the recogniser learns it as a token.
    """
    rng = np.random.default_rng(seed)
    params = params or random_session_params(rng, mount_spread_deg=mount_spread_deg, mode=mode)
    script = calibration_script(rng) if calibrate else [ScriptItem("gap", rest(2.0))]
    for text in texts:
        if clench_rate > 0 and rng.random() < clench_rate:
            text = f"{text} <clench>"
        kind = "sentence" if " " in text else "word"
        script.append(utterance_item(text, lexicon, rng, kind=kind, tissue_gain=tissue_gain))
    return render_session(script, params, rng, utt_prefix=utt_prefix)


__all__ = [
    "MODE_FIFO",
    "MODE_POLLED",
    "ScriptItem",
    "SessionParams",
    "SimSession",
    "calibration_script",
    "random_session_params",
    "render_session",
    "simulate_session",
    "utterance_item",
]
