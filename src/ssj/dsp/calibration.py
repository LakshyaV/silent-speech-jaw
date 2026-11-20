"""Session calibration: put both IMUs into one anatomical frame.

The two sensors are stuck to the skin at arbitrary orientations and are
re-mounted for every session, so raw axes mean something different every time
the device is worn. A ~20 s routine fixes this physically instead of asking the
network to memorise mounts:

``rest``  jaw closed, head still
          -> gyro biases and the gravity vector seen by each sensor
``head``  nod and shake the head with the jaw relaxed
          -> both sensors sense the same rigid rotation, each in its own frame;
             solving Wahba's problem on the gyro pairs gives the rotation
             ``R_ref_to_jaw`` that expresses the reference IMU in the jaw frame
``jaw``   open and close a few times with the head still
          -> the principal axis of the *relative* rotation is the mandibular
             hinge axis

Hinge axis and gravity define a canonical frame fixed to the anatomy:

    x  hinge axis, signed so that opening the jaw is a positive rotation
    z  up (away from gravity), orthogonal to x
    y  completes the right-handed frame (z x x)

In this frame "channel 0 of the relative gyro" is jaw opening velocity for every
session and every wearer, and ``jaw - ref`` cancels head motion exactly rather
than approximately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..io.protocol import SENSOR_JAW, SENSOR_REF, split_sensors, to_physical
from ..io.session import Session
from .resample import DualResampler

_EYE = np.eye(3)


def _unit(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        return np.asarray(fallback if fallback is not None else [0.0, 0.0, 1.0], dtype=np.float64)
    return np.asarray(v, dtype=np.float64) / norm


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking direction ``a`` onto direction ``b`` (Rodrigues)."""
    a, b = _unit(a), _unit(b)
    axis = np.cross(a, b)
    s, c = float(np.linalg.norm(axis)), float(np.dot(a, b))
    if s < 1e-9:
        if c > 0:
            return _EYE.copy()
        # 180 degrees: rotate about any axis orthogonal to ``a``
        ortho = _unit(np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0]))
        return 2.0 * np.outer(ortho, ortho) - _EYE
    k = axis / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return _EYE + s * K + (1 - c) * (K @ K)


def kabsch(source: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Rotation ``R`` minimising ``sum w_i |target_i - R source_i|^2`` (proper, det = +1)."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if weights is not None:
        source = source * np.asarray(weights, dtype=np.float64)[:, None]
    H = source.T @ target
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def angle_between_rotations(A: np.ndarray, B: np.ndarray) -> float:
    """Geodesic distance between two rotations, in degrees."""
    cos = (np.trace(A.T @ B) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _robust_sigma(x: np.ndarray) -> np.ndarray:
    median = np.median(x, axis=0)
    return 1.4826 * np.median(np.abs(x - median), axis=0)


@dataclass
class Calibration:
    """Per-session alignment and bias parameters applied by the front-end."""

    bias_jaw: np.ndarray = field(default_factory=lambda: np.zeros(3))  # deg/s, jaw sensor frame
    bias_ref: np.ndarray = field(default_factory=lambda: np.zeros(3))  # deg/s, ref sensor frame
    g_jaw: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))  # g at rest
    g_ref: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    R_ref_to_jaw: np.ndarray = field(default_factory=lambda: np.eye(3))
    R_canon: np.ndarray = field(default_factory=lambda: np.eye(3))  # jaw sensor frame -> anatomical
    noise_gyro_dps: float = 0.0  # measured relative-rate noise at rest (0 = unknown)
    noise_accel_g: float = 0.0
    quality: dict[str, Any] = field(default_factory=dict)

    @property
    def R_jaw(self) -> np.ndarray:
        """Jaw sensor frame -> canonical frame."""
        return self.R_canon

    @property
    def R_ref(self) -> np.ndarray:
        """Reference sensor frame -> canonical frame."""
        return self.R_canon @ self.R_ref_to_jaw

    @property
    def level(self) -> str:
        """``full`` (all three phases), ``partial`` or ``none``."""
        return str(self.quality.get("level", "none"))

    @classmethod
    def identity(cls) -> Calibration:
        return cls(quality={"level": "none"})

    def to_dict(self) -> dict[str, Any]:
        return {
            "bias_jaw": self.bias_jaw.tolist(),
            "bias_ref": self.bias_ref.tolist(),
            "g_jaw": self.g_jaw.tolist(),
            "g_ref": self.g_ref.tolist(),
            "R_ref_to_jaw": self.R_ref_to_jaw.tolist(),
            "R_canon": self.R_canon.tolist(),
            "noise_gyro_dps": float(self.noise_gyro_dps),
            "noise_accel_g": float(self.noise_accel_g),
            "quality": self.quality,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Calibration:
        return cls(
            bias_jaw=np.asarray(data["bias_jaw"], dtype=np.float64),
            bias_ref=np.asarray(data["bias_ref"], dtype=np.float64),
            g_jaw=np.asarray(data["g_jaw"], dtype=np.float64),
            g_ref=np.asarray(data["g_ref"], dtype=np.float64),
            R_ref_to_jaw=np.asarray(data["R_ref_to_jaw"], dtype=np.float64),
            R_canon=np.asarray(data["R_canon"], dtype=np.float64),
            noise_gyro_dps=float(data.get("noise_gyro_dps", 0.0)),
            noise_accel_g=float(data.get("noise_accel_g", 0.0)),
            quality=dict(data.get("quality") or {}),
        )


def estimate_ref_to_jaw(
    gyro_jaw: np.ndarray,
    gyro_ref: np.ndarray,
    g_jaw: np.ndarray,
    g_ref: np.ndarray,
    *,
    min_rate_dps: float = 4.0,
    gravity_weight: float = 0.5,
) -> tuple[np.ndarray, dict[str, float]]:
    """Solve for the rotation that maps reference-frame vectors into the jaw frame.

    The gravity pair is always included as one extra observation: it is valid
    whenever the jaw is closed and it pins the solution when the head motion
    only excited one or two axes.
    """
    rate = np.linalg.norm(gyro_ref, axis=1)
    moving = rate > min_rate_dps
    source = [_unit(g_ref)[None, :]]
    target = [_unit(g_jaw)[None, :]]
    energy = float(np.sum(gyro_ref[moving] ** 2)) if moving.any() else 0.0
    weights = [np.array([np.sqrt(max(gravity_weight * energy, 1.0))])]
    if moving.any():
        source.append(gyro_ref[moving])
        target.append(gyro_jaw[moving])
        weights.append(np.ones(int(moving.sum())))
    S, T, W = np.concatenate(source), np.concatenate(target), np.concatenate(weights)
    R = kabsch(S * W[:, None], T * W[:, None])

    report = {"head_samples": int(moving.sum())}
    if moving.sum() >= 3:
        residual = gyro_jaw[moving] - gyro_ref[moving] @ R.T
        report["head_residual"] = float(np.linalg.norm(residual) / max(np.linalg.norm(gyro_jaw[moving]), 1e-9))
        singular = np.linalg.svd(gyro_ref[moving], compute_uv=False)
        report["head_excitation"] = float(singular[1] / max(singular[0], 1e-9))
    return R, report


def estimate_hinge_axis(rel_gyro: np.ndarray) -> tuple[np.ndarray, float]:
    """Principal axis of the relative rotation and the variance it explains.

    The sign is chosen so that the first clear excursion is positive: the
    calibration protocol starts with the mouth closed and opens first.
    """
    covariance = rel_gyro.T @ rel_gyro
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, -1]
    explained = float(eigenvalues[-1] / max(eigenvalues.sum(), 1e-12))
    projection = rel_gyro @ axis
    peak = float(np.max(np.abs(projection))) if len(projection) else 0.0
    if peak > 0:
        first = int(np.argmax(np.abs(projection) > 0.3 * peak))
        # Average a short window after onset so one noisy sample cannot flip the sign.
        window = projection[first : first + max(3, len(projection) // 200)]
        if np.sum(window) < 0:
            axis = -axis
    return _unit(axis), explained


def canonical_frame(hinge_axis: np.ndarray | None, up: np.ndarray) -> np.ndarray:
    """Rotation from the jaw sensor frame to the anatomical frame (rows are x, y, z)."""
    up = _unit(up)
    if hinge_axis is None:
        return rotation_between(up, np.array([0.0, 0.0, 1.0]))
    x = _unit(hinge_axis)
    z = up - np.dot(up, x) * x
    if np.linalg.norm(z) < 0.2:  # hinge nearly vertical: the head was not upright
        return rotation_between(up, np.array([0.0, 0.0, 1.0]))
    z = _unit(z)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def calibrate(
    rest: np.ndarray | None,
    head: np.ndarray | None = None,
    jaw: np.ndarray | None = None,
) -> Calibration:
    """Estimate a :class:`Calibration` from uniformly sampled segments.

    Each segment is ``[n, 12]``: jaw accel, jaw gyro, ref accel, ref gyro
    (g and deg/s). Missing segments degrade gracefully: without ``head`` the two
    frames are aligned by gravity alone, without ``jaw`` the canonical frame is
    only levelled.
    """
    if rest is None or len(rest) < 8:
        return Calibration.identity()
    cal = Calibration()
    cal.bias_jaw = rest[:, 3:6].mean(axis=0)
    cal.bias_ref = rest[:, 9:12].mean(axis=0)
    cal.g_jaw = rest[:, 0:3].mean(axis=0)
    cal.g_ref = rest[:, 6:9].mean(axis=0)
    quality: dict[str, Any] = {"level": "partial", "rest_samples": len(rest)}

    if head is not None and len(head) >= 16:
        cal.R_ref_to_jaw, report = estimate_ref_to_jaw(
            head[:, 3:6] - cal.bias_jaw, head[:, 9:12] - cal.bias_ref, cal.g_jaw, cal.g_ref
        )
        quality.update(report)
        quality["alignment"] = "gyro+gravity" if report["head_samples"] >= 16 else "gravity"
    else:
        cal.R_ref_to_jaw = rotation_between(cal.g_ref, cal.g_jaw)
        quality["alignment"] = "gravity"

    hinge = None
    if jaw is not None and len(jaw) >= 16:
        relative = (jaw[:, 3:6] - cal.bias_jaw) - (jaw[:, 9:12] - cal.bias_ref) @ cal.R_ref_to_jaw.T
        if np.sqrt(np.mean(relative**2)) > 1.0:  # the jaw actually moved
            hinge, explained = estimate_hinge_axis(relative)
            quality["hinge_explained"] = explained
    cal.R_canon = canonical_frame(hinge, cal.g_jaw)
    if hinge is not None and quality["alignment"] == "gyro+gravity":
        quality["level"] = "full"

    rel_gyro = (rest[:, 3:6] - cal.bias_jaw) - (rest[:, 9:12] - cal.bias_ref) @ cal.R_ref_to_jaw.T
    rel_accel = rest[:, 0:3] - rest[:, 6:9] @ cal.R_ref_to_jaw.T
    cal.noise_gyro_dps = float(np.sqrt(np.mean(_robust_sigma(rel_gyro) ** 2)))
    cal.noise_accel_g = float(np.sqrt(np.mean(_robust_sigma(rel_accel) ** 2)))
    cal.quality = quality
    return cal


def paired_stream(raw: np.ndarray, info, fs: float) -> tuple[int, np.ndarray]:
    """Resample a raw record to uniform ``[n, 12]`` physical rows; returns ``(k0, rows)``."""
    resampler = DualResampler(fs)
    parts = split_sensors(raw)
    for sensor in (SENSOR_JAW, SENSOR_REF):
        resampler.push(sensor, *to_physical(parts[sensor], info))
    k0, rows, _ = resampler.pop()
    return k0, rows


def calibrate_session(session: Session, fs: float = 200.0, *, fallback_rest_s: float = 1.5) -> Calibration:
    """Calibrate from a session's recorded calibration phases.

    A session without calibration events still gets gyro biases and a gravity
    alignment from its first ``fallback_rest_s`` seconds, which are assumed still.
    """
    segments = session.calibration_segments()
    raw = session.raw()
    if len(raw) == 0:
        return Calibration.identity()
    if segments:
        t_end = max(end for _, end in segments.values())
        raw = raw[raw["t_us"] <= t_end + 200_000]
    else:
        raw = raw[raw["t_us"] <= raw["t_us"].min() + int(fallback_rest_s * 1e6)]
    k0, rows = paired_stream(raw, session.info, fs)
    if len(rows) == 0:
        return Calibration.identity()

    def cut(phase: str) -> np.ndarray | None:
        if phase not in segments:
            return None
        start, end = segments[phase]
        a = max(int(np.ceil(start * 1e-6 * fs)) - k0, 0)
        b = min(int(np.floor(end * 1e-6 * fs)) - k0, len(rows))
        return rows[a:b] if b - a > 0 else None

    if not segments:
        calibration = calibrate(rows)
        calibration.quality["source"] = "leading-rest"
        return calibration
    calibration = calibrate(cut("rest"), cut("head"), cut("jaw"))
    calibration.quality["source"] = "routine"
    return calibration
