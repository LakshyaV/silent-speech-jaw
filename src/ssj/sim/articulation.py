"""Phoneme sequence -> mandible trajectory.

A small articulatory model in the spirit of task dynamics: every phoneme sets a
target for the jaw (opening angle about the hinge, protrusion), targets are
blended according to how strongly each phoneme constrains the jaw, and a
critically damped second-order system moves the jaw between them, which yields
coarticulation and target undershoot at fast rates for free.

The model is deliberately honest about what a jaw sensor can and cannot see:
voiced/voiceless pairs (P/B, T/D, K/G, F/V, S/Z ...) have *identical* targets,
nasals differ from their stops only slightly in duration, and velars and /h/
barely constrain the jaw at all. Data rendered from it therefore carries the
same homophene ambiguity as the real signal, which is what makes it useful for
pre-training the recogniser and for exercising the decoder.

Besides the rigid jaw motion, a skin-mounted sensor is tilted slightly by the
tissue under it as the tongue and the floor of the mouth move. This is modelled
as a small two-axis tilt that depends on place of articulation (``TISSUE``),
identical for all phonemes of an articulatory class. It encodes the working
assumption behind the class inventory - that place is partly observable - and
its strength is a parameter (``tissue_gain``; 0 removes it).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class PhoneSpec:
    opening: tuple[float, ...]  # target opening angle(s) in degrees; two values = diphthong
    protrusion: tuple[float, ...]  # target protrusion in mm (negative = retracted)
    weight: float  # 0..1, how strongly the phoneme constrains the jaw
    duration_ms: float
    burst: bool = False  # stop release transient
    tissue: tuple[float, float] = (0.0, 0.0)  # soft-tissue tilt (deg) about the two non-hinge axes


def _v(opening, protrusion=0.0, duration=150.0, weight=1.0) -> PhoneSpec:
    o = opening if isinstance(opening, tuple) else (opening,)
    p = protrusion if isinstance(protrusion, tuple) else (protrusion,) * len(o)
    return PhoneSpec(tuple(map(float, o)), tuple(map(float, p)), weight, duration)


def _c(opening, protrusion=0.0, duration=85.0, weight=0.8, burst=False) -> PhoneSpec:
    return PhoneSpec((float(opening),), (float(protrusion),), weight, duration, burst)


_BILABIAL = _c(0.8, 0.0, 85, 1.0, burst=True)
_ALVEOLAR_STOP = _c(2.0, 0.0, 75, 0.8, burst=True)
_VELAR_STOP = _c(3.4, 0.0, 80, 0.25, burst=True)
_LABIODENTAL = _c(1.6, -0.8, 105, 0.9)
_DENTAL = _c(2.4, 0.2, 95, 0.7)
_SIBILANT = _c(1.1, 0.0, 115, 1.0)
_POSTALVEOLAR = _c(1.6, 1.0, 115, 0.95)
_AFFRICATE = _c(1.6, 1.0, 125, 0.95, burst=True)

#: Articulatory targets. Voicing pairs share one spec object on purpose.
PHONE_SPECS: dict[str, PhoneSpec] = {
    # vowels: opening scales with vowel height, protrusion with rounding
    "IY": _v(2.8, 0.0, 135), "IH": _v(4.0, 0.0, 115),
    "UW": _v(3.0, 2.0, 140), "UH": _v(4.2, 1.2, 115),
    "EH": _v(6.5, 0.0, 130), "EY": _v((6.2, 3.4), 0.0, 185),
    "AH": _v(5.5, 0.0, 110), "ER": _v(4.6, 0.6, 150),
    "OW": _v((6.6, 4.0), (1.0, 1.8), 185), "AO": _v(8.6, 1.0, 160),
    "AA": _v(11.0, 0.0, 165), "AE": _v(10.0, 0.0, 165),
    "AY": _v((10.2, 3.4), 0.0, 215), "AW": _v((10.0, 3.8), (0.0, 1.8), 215),
    "OY": _v((7.6, 3.4), (1.6, 0.0), 220),
    # consonants
    "P": _BILABIAL, "B": _BILABIAL, "M": _c(0.8, 0.0, 95, 1.0),
    "F": _LABIODENTAL, "V": _LABIODENTAL,
    "TH": _DENTAL, "DH": _DENTAL,
    "T": _ALVEOLAR_STOP, "D": _ALVEOLAR_STOP, "N": _c(2.0, 0.0, 80, 0.8),
    "S": _SIBILANT, "Z": _SIBILANT,
    "SH": _POSTALVEOLAR, "ZH": _POSTALVEOLAR,
    "CH": _AFFRICATE, "JH": _AFFRICATE,
    "L": _c(3.2, 0.0, 75, 0.5), "R": _c(3.0, 0.8, 80, 0.5),
    "K": _VELAR_STOP, "G": _VELAR_STOP, "NG": _c(3.4, 0.0, 85, 0.25),
    "HH": _c(3.0, 0.0, 60, 0.05),
    "W": _c(2.5, 2.0, 75, 0.6), "Y": _c(2.6, 0.0, 70, 0.6),
}  # fmt: skip

#: Soft-tissue tilt by place of articulation (degrees about y and z). Consonants of
#: one class share a value exactly; vowels vary with tongue advancement.
TISSUE: dict[str, tuple[float, float]] = {
    "IY": (0.5, 0.3), "IH": (0.4, 0.25), "EH": (0.3, 0.15), "EY": (0.35, 0.2), "AE": (0.2, 0.0),
    "AH": (0.0, 0.0), "ER": (-0.1, 0.5), "AA": (-0.3, -0.1), "AO": (-0.5, 0.0), "OW": (-0.5, 0.2),
    "UH": (-0.5, 0.3), "UW": (-0.6, 0.4), "AY": (0.0, 0.1), "AW": (-0.3, 0.2), "OY": (-0.3, 0.3),
    "P": (0.0, 0.0), "B": (0.0, 0.0), "M": (0.0, 0.0),
    "F": (0.2, -0.3), "V": (0.2, -0.3),
    "TH": (0.5, 0.1), "DH": (0.5, 0.1),
    "T": (0.8, 0.0), "D": (0.8, 0.0), "N": (0.8, 0.0),
    "S": (0.7, 0.4), "Z": (0.7, 0.4),
    "SH": (0.5, 0.8), "ZH": (0.5, 0.8), "CH": (0.5, 0.8), "JH": (0.5, 0.8),
    "L": (0.9, -0.5), "R": (-0.2, 0.7),
    "K": (-0.8, 0.3), "G": (-0.8, 0.3), "NG": (-0.8, 0.3),
    "HH": (0.0, 0.0), "W": (-0.5, 0.6), "Y": (0.4, 0.9),
}  # fmt: skip
PHONE_SPECS = {name: replace(spec, tissue=TISSUE[name]) for name, spec in PHONE_SPECS.items()}

CLENCH_DURATION_S = 0.32


@dataclass
class Trajectory:
    """Jaw motion relative to the skull, sampled at ``fs``."""

    fs: float
    opening_deg: np.ndarray
    protrusion_mm: np.ndarray
    lateral_deg: np.ndarray
    tissue_deg: np.ndarray  # [n, 2] soft-tissue tilt about the two non-hinge axes
    vibration_g: np.ndarray  # [n, 3] high-frequency skin vibration (bursts, clench)
    phone_times: list[tuple[str, float, float]]  # (phoneme, start_s, end_s)

    @property
    def duration_s(self) -> float:
        return len(self.opening_deg) / self.fs

    def __len__(self) -> int:
        return len(self.opening_deg)


def _burst(
    n: int, fs: float, rng: np.random.Generator, amplitude_g: float, freq_hz: float, decay_s: float
) -> np.ndarray:
    t = np.arange(n) / fs
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    wave = amplitude_g * np.exp(-t / decay_s) * np.sin(2 * np.pi * freq_hz * t + rng.uniform(0, 2 * np.pi))
    return wave[:, None] * direction[None, :]


def synthesize(
    phonemes: list[str],
    rng: np.random.Generator,
    *,
    fs: float = 400.0,
    rate: float | None = None,
    word_boundaries: list[int] | None = None,
    effort: float | None = None,
    tissue_gain: float = 1.0,
) -> Trajectory:
    """Render one utterance, starting and ending in the rest posture.

    ``rate`` > 1 is faster speech. ``word_boundaries`` are indices into
    ``phonemes`` where a word starts; short pauses are occasionally inserted
    there. ``effort`` scales articulation amplitude (hyper- vs hypo-articulation).
    """
    rate = float(rng.uniform(0.8, 1.3)) if rate is None else rate
    effort = float(rng.uniform(0.8, 1.2)) if effort is None else effort
    lead, tail = 0.12, 0.25  # rest before and after, seconds
    boundaries = set(word_boundaries or [])

    # 1. Piecewise targets -------------------------------------------------
    times: list[float] = [0.0]  # segment start times
    opening: list[float] = [0.0]
    protrusion: list[float] = [0.0]
    tissue: list[tuple[float, float]] = [(0.0, 0.0)]
    weights: list[float] = [1.0]
    bursts: list[float] = []
    clenches: list[float] = []
    phone_times: list[tuple[str, float, float]] = []
    cursor = lead

    for index, phoneme in enumerate(phonemes):
        if index in boundaries and index > 0 and rng.random() < 0.15:
            pause = float(rng.uniform(0.08, 0.22))
            times.append(cursor)
            opening.append(0.6)
            protrusion.append(0.0)
            tissue.append((0.0, 0.0))
            weights.append(0.6)
            cursor += pause
        if phoneme == "CLENCH":
            times.append(cursor)
            opening.append(-0.5)  # teeth pressed together, slightly past rest
            protrusion.append(0.0)
            tissue.append((0.0, 0.0))
            weights.append(1.0)
            clenches.append(cursor)
            phone_times.append((phoneme, cursor, cursor + CLENCH_DURATION_S))
            cursor += CLENCH_DURATION_S
            continue
        spec = PHONE_SPECS[phoneme]
        duration = spec.duration_ms * 1e-3 / rate * float(np.exp(rng.normal(0.0, 0.16)))
        start = cursor
        parts = len(spec.opening)
        splits = [0.58, 0.42] if parts == 2 else [1.0]
        for part in range(parts):
            times.append(cursor)
            jitter = float(np.exp(rng.normal(0.0, 0.07)))
            opening.append(spec.opening[part] * effort * jitter)
            protrusion.append(spec.protrusion[part] * effort)
            tissue.append((spec.tissue[0] * effort, spec.tissue[1] * effort))
            weights.append(spec.weight)
            cursor += duration * splits[part]
        if spec.burst:
            bursts.append(cursor)
        phone_times.append((phoneme, start, cursor))
    times.append(cursor)  # return to rest
    opening.append(0.0)
    protrusion.append(0.0)
    tissue.append((0.0, 0.0))
    weights.append(1.0)
    total = cursor + tail

    # 2. Weak phonemes inherit their neighbours' targets (coarticulation) ----
    op = np.asarray(opening)
    pr = np.asarray(protrusion)
    ti = np.asarray(tissue) * tissue_gain
    w = np.asarray(weights)
    for i in range(1, len(op) - 1):
        if w[i] < 1.0:  # only the rigid jaw is inherited; the tongue still does its own thing
            neighbour_o = 0.35 * op[i - 1] + 0.65 * op[i + 1]
            neighbour_p = 0.35 * pr[i - 1] + 0.65 * pr[i + 1]
            op[i] = w[i] * op[i] + (1 - w[i]) * neighbour_o
            pr[i] = w[i] * pr[i] + (1 - w[i]) * neighbour_p

    # 3. Critically damped second-order tracking ---------------------------
    n = int(np.ceil(total * fs))
    grid = np.arange(n) / fs
    segment = np.clip(np.searchsorted(np.asarray(times), grid, side="right") - 1, 0, len(op) - 1)
    # Four cascaded real poles: a smooth, bell-shaped velocity profile like a real
    # jaw gesture (a plain second-order step response starts with a jerk).
    pole_hz = float(rng.uniform(9.0, 13.0))
    a = float(np.exp(-2 * np.pi * pole_hz / fs))
    sos = np.tile(np.array([(1 - a) ** 2, 0.0, 0.0, 1.0, -2 * a, a * a]), (2, 1))
    opening_track = signal.sosfilt(sos, op[segment])
    protrusion_track = signal.sosfilt(sos, pr[segment])
    tissue_track = signal.sosfilt(sos, ti[segment], axis=0)
    wobble = signal.sosfilt(sos, rng.normal(0.0, 1.0, n))
    lateral = wobble * 6.0 * (np.abs(np.gradient(opening_track)) * fs / 60.0)

    # 4. Skin vibration: stop releases and clenches ------------------------
    vibration = np.zeros((n, 3))
    for t_burst in bursts:
        i0 = int(t_burst * fs)
        length = min(int(0.05 * fs), n - i0)
        if length > 4:
            vibration[i0 : i0 + length] += _burst(length, fs, rng, rng.uniform(0.012, 0.03), rng.uniform(55, 85), 0.012)
    for t_clench in clenches:
        i0 = int(t_clench * fs)
        length = min(int(CLENCH_DURATION_S * fs), n - i0)
        if length > 8:
            noise = rng.normal(0.0, 1.0, (length, 3))
            sos = signal.butter(2, [45, min(95, 0.45 * fs)], "bandpass", fs=fs, output="sos")
            window = np.hanning(length)[:, None]
            vibration[i0 : i0 + length] += signal.sosfilt(sos, noise, axis=0) * window * rng.uniform(0.02, 0.04)

    return Trajectory(fs, opening_track, protrusion_track, lateral, tissue_track, vibration, phone_times)


def rest(duration_s: float, fs: float = 400.0) -> Trajectory:
    n = round(duration_s * fs)
    zeros = np.zeros(n)
    return Trajectory(fs, zeros, zeros.copy(), zeros.copy(), np.zeros((n, 2)), np.zeros((n, 3)), [])


def open_close(
    cycles: int, rng: np.random.Generator, fs: float = 400.0, amplitude_deg: float = 12.0, period_s: float = 1.1
) -> Trajectory:
    """Calibration movement: slow open/close cycles starting from closed."""
    n = int(cycles * period_s * fs)
    t = np.arange(n) / fs
    amplitude = amplitude_deg * rng.uniform(0.85, 1.15)
    opening = amplitude * 0.5 * (1 - np.cos(2 * np.pi * t / period_s))
    zeros = np.zeros(n)
    return Trajectory(fs, opening, zeros, zeros.copy(), np.zeros((n, 2)), np.zeros((n, 3)), [])
