"""Physically motivated augmentation of front-end features.

Recorded data is scarce and every nuisance below is one the system will meet in
use. Wherever possible the augmentation follows the physics of the sensor pair
rather than treating channels as anonymous numbers:

* **rate**: speech tempo varies, so time is stretched and angular rates are
  scaled inversely, as they would be for the same movement made slower.
* **head motion**: a random rigid rotation is added identically to the jaw and
  reference streams (both live in the canonical frame), with the gravity shift it
  implies on both accelerometers. The relative stream is untouched, which
  teaches the network that common-mode motion carries no speech.
* **frame error**: all 3-vectors are rotated by a few degrees, the residual a
  real calibration leaves behind.
* **gain, noise, dropout, masking**: sensor coupling changes, an axis goes noisy,
  a few frames are lost.

Features are augmented in front-end units, before dataset normalisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dsp.frontend import FEATURE_GROUPS, GYRO_TRIPLETS, VECTOR_TRIPLETS, FrontendConfig


@dataclass(frozen=True)
class AugmentConfig:
    enabled: bool = True
    rate_range: tuple[float, float] = (0.82, 1.22)  # time-stretch factor (>1 = slower)
    gain_range: tuple[float, float] = (0.85, 1.18)  # per stream group
    rotation_deg: float = 6.0  # residual calibration error
    head_motion_prob: float = 0.5
    head_rate_dps: tuple[float, float] = (2.0, 25.0)
    noise_std: float = 0.03  # relative to each channel's RMS
    drift_prob: float = 0.3  # slow accelerometer baseline drift
    time_masks: int = 2
    time_mask_frames: int = 6
    stream_dropout: float = 0.05  # probability of zeroing a whole stream group
    concat_prob: float = 0.15  # join two utterances of the same session


def _rotation(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-12
    angle = np.radians(rng.uniform(-max_deg, max_deg))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _smooth(rng: np.random.Generator, n: int, channels: int, cutoff: float) -> np.ndarray:
    """Band-limited unit-RMS noise via a moving average (cutoff in cycles per frame)."""
    width = max(3, int(0.44 / max(cutoff, 1e-3)))
    raw = rng.normal(size=(n + 2 * width, channels))
    kernel = np.hanning(width)
    kernel /= kernel.sum()
    out = np.stack([np.convolve(raw[:, c], kernel, mode="same") for c in range(channels)], axis=1)[width:-width]
    return out / (out.std(axis=0, keepdims=True) + 1e-12)


def time_stretch(feats: np.ndarray, factor: float) -> np.ndarray:
    n = len(feats)
    m = max(2, round(n * factor))
    src = np.linspace(0.0, n - 1.0, m)
    left = np.floor(src).astype(int)
    right = np.minimum(left + 1, n - 1)
    weight = (src - left)[:, None]
    out = feats[left] * (1 - weight) + feats[right] * weight
    for start in GYRO_TRIPLETS:  # the same movement made slower is a lower angular rate
        out[:, start : start + 3] /= factor
    return out.astype(np.float32)


def inject_head_motion(
    feats: np.ndarray, rng: np.random.Generator, rate_dps: float, frontend: FrontendConfig
) -> np.ndarray:
    n = len(feats)
    omega = _smooth(rng, n, 3, 1.5 / frontend.fs) * rate_dps  # deg/s, canonical frame
    angle = np.radians(np.cumsum(omega, axis=0) / frontend.fs)
    angle -= angle.mean(axis=0, keepdims=True)
    gravity_shift = np.cross(angle, np.array([0.0, 0.0, 1.0]))  # g, small-angle
    out = feats.copy()
    for group in ("jaw", "ref"):
        a, _ = FEATURE_GROUPS[group]
        out[:, a : a + 3] += (omega / frontend.gyro_scale_dps).astype(np.float32)
        out[:, a + 3 : a + 6] += (gravity_shift / frontend.accel_scale_g).astype(np.float32)
    return out


def augment(
    feats: np.ndarray, rng: np.random.Generator, config: AugmentConfig, frontend: FrontendConfig | None = None
) -> np.ndarray:
    """Return an augmented copy of one utterance's features ``[T, C]``."""
    if not config.enabled:
        return feats
    frontend = frontend or FrontendConfig()
    out = time_stretch(feats, float(rng.uniform(*config.rate_range)))
    n = len(out)

    if config.rotation_deg > 0:
        R = _rotation(rng, config.rotation_deg).astype(np.float32)
        for start in VECTOR_TRIPLETS:
            out[:, start : start + 3] = out[:, start : start + 3] @ R.T
    if rng.random() < config.head_motion_prob:
        out = inject_head_motion(out, rng, float(rng.uniform(*config.head_rate_dps)), frontend)
    for group, (a, b) in FEATURE_GROUPS.items():
        if group == "env":
            continue
        out[:, a:b] *= float(rng.uniform(*config.gain_range))
    if rng.random() < config.drift_prob:
        drift = _smooth(rng, n, 3, 0.4 / frontend.fs) * rng.uniform(0.0, 0.08)
        for group in ("jaw", "ref"):
            a, _ = FEATURE_GROUPS[group]
            out[:, a + 3 : a + 6] += drift.astype(np.float32)
    if config.noise_std > 0:
        rms = np.sqrt(np.mean(out**2, axis=0, keepdims=True)) + 1e-6
        out += (rng.normal(size=out.shape) * rms * rng.uniform(0, config.noise_std)).astype(np.float32)
    for _ in range(config.time_masks):
        width = int(rng.integers(0, config.time_mask_frames + 1))
        if width and n > 4 * width:
            start = int(rng.integers(0, n - width))
            out[start : start + width] = 0.0
    if config.stream_dropout > 0 and rng.random() < config.stream_dropout:
        a, b = FEATURE_GROUPS[str(rng.choice(["jaw", "ref", "env"]))]
        out[:, a:b] = 0.0
    return out.astype(np.float32)
