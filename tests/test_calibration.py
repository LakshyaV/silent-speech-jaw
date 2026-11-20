from __future__ import annotations

import numpy as np
import pytest

from ssj.dsp.calibration import (
    Calibration,
    angle_between_rotations,
    calibrate,
    calibrate_session,
    canonical_frame,
    estimate_hinge_axis,
    kabsch,
    paired_stream,
    rotation_between,
)
from ssj.io import Session
from ssj.sim import simulate_session
from ssj.sim.imu import MODE_POLLED, random_rotation


def test_kabsch_recovers_a_known_rotation():
    rng = np.random.default_rng(0)
    R = random_rotation(rng)
    source = rng.normal(size=(200, 3))
    target = source @ R.T + rng.normal(scale=0.01, size=(200, 3))
    estimate = kabsch(source, target)
    assert angle_between_rotations(estimate, R) < 0.5
    assert np.linalg.det(estimate) == pytest.approx(1.0)


def test_rotation_between_handles_all_cases():
    rng = np.random.default_rng(1)
    for _ in range(20):
        a, b = rng.normal(size=3), rng.normal(size=3)
        R = rotation_between(a, b)
        assert np.allclose(R @ (a / np.linalg.norm(a)), b / np.linalg.norm(b), atol=1e-9)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.allclose(rotation_between([0, 0, 1], [0, 0, 1]), np.eye(3))
    flip = rotation_between([0, 0, 1], [0, 0, -1])
    assert np.allclose(flip @ [0, 0, 1], [0, 0, -1])


def test_hinge_axis_sign_follows_the_first_movement():
    t = np.linspace(0, 4, 800)
    axis = np.array([0.6, 0.0, 0.8])
    opening_rate = np.sin(2 * np.pi * t)  # starts by opening
    estimate, explained = estimate_hinge_axis(np.outer(opening_rate, axis))
    assert np.allclose(estimate, axis, atol=1e-6) and explained > 0.99
    estimate, _ = estimate_hinge_axis(np.outer(opening_rate, -axis))
    assert np.allclose(estimate, -axis, atol=1e-6)


def test_canonical_frame_is_right_handed_and_levelled():
    R = canonical_frame(np.array([1.0, 0.1, 0.0]), np.array([0.0, 0.2, 1.0]))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.linalg.det(R) == pytest.approx(1.0)
    up = np.array([0.0, 0.2, 1.0]) / np.linalg.norm([0.0, 0.2, 1.0])
    assert (R @ up)[2] > 0.95  # gravity ends up along +z
    assert np.allclose(R @ (np.array([1.0, 0.1, 0.0]) / np.linalg.norm([1.0, 0.1, 0.0])), [1, 0, 0], atol=1e-9)
    assert (canonical_frame(None, up) @ up)[2] == pytest.approx(1.0)


@pytest.mark.parametrize("seed,mode", [(1, 1), (2, 1), (5, MODE_POLLED)])
def test_calibration_recovers_arbitrary_mounts(tmp_path, lexicon, seed, mode):
    """Both sensors are mounted at random orientations; calibration must undo that."""
    sim = simulate_session(["hello how are you today"], lexicon, seed=seed, mode=mode)
    sim.write(tmp_path / "s", subject="sim", session="s")
    calibration = calibrate_session(Session(tmp_path / "s"))
    assert calibration.level == "full"
    truth = sim.truth
    assert angle_between_rotations(calibration.R_ref_to_jaw, np.array(truth["R_ref_to_jaw"])) < 1.0
    hinge = np.array(truth["hinge_axis_jaw_sensor"])
    assert np.degrees(np.arccos(np.clip(hinge @ calibration.R_canon[0], -1, 1))) < 3.0
    assert angle_between_rotations(calibration.R_canon, np.array(truth["R_canon"])) < 3.0
    # The head sways during the rest phase, so each bias is only known to within that
    # sway; what matters is that the *relative* rate (jaw minus aligned reference) is unbiased.
    error_jaw = calibration.bias_jaw - sim.params.bias_jaw_dps
    error_ref = calibration.bias_ref - sim.params.bias_ref_dps
    assert np.abs(error_jaw).max() < 1.5
    assert np.allclose(error_jaw, calibration.R_ref_to_jaw @ error_ref, atol=0.2)
    assert calibration.quality["head_residual"] < 0.1
    assert calibration.quality["hinge_explained"] > 0.95
    assert 0 < calibration.noise_gyro_dps < 1.0
    assert Calibration.from_dict(calibration.to_dict()).R_canon.tolist() == calibration.R_canon.tolist()


def test_partial_calibration_degrades_gracefully(sim_session):
    k0, rows = paired_stream(sim_session.raw, sim_session.info, 200.0)
    segments = {e["phase"]: (e["t_start_us"], e["t_end_us"]) for e in sim_session.events if e["type"] == "calibration"}

    def cut(phase):
        a, b = (max(int(v * 1e-6 * 200) - k0, 0) for v in segments[phase])
        return rows[a:b]

    assert calibrate(None).level == "none"
    rest_only = calibrate(cut("rest"))
    assert rest_only.level == "partial" and rest_only.quality["alignment"] == "gravity"
    # Gravity alone still aligns the "up" direction of the two sensors.
    assert np.allclose(
        rest_only.R_ref_to_jaw @ (rest_only.g_ref / np.linalg.norm(rest_only.g_ref)),
        rest_only.g_jaw / np.linalg.norm(rest_only.g_jaw),
        atol=1e-6,
    )
    assert (rest_only.R_canon @ rest_only.g_jaw)[2] == pytest.approx(np.linalg.norm(rest_only.g_jaw), rel=1e-6)
    no_jaw = calibrate(cut("rest"), cut("head"))
    assert no_jaw.level == "partial" and no_jaw.quality["alignment"] == "gyro+gravity"
    assert calibrate(cut("rest"), cut("head"), cut("jaw")).level == "full"


def test_session_without_routine_falls_back_to_leading_rest(tmp_path, lexicon):
    sim = simulate_session(["good morning everyone"], lexicon, seed=4, calibrate=False)
    sim.write(tmp_path / "s", subject="sim", session="s")
    calibration = calibrate_session(Session(tmp_path / "s"))
    assert calibration.level == "partial" and calibration.quality["source"] == "leading-rest"
    assert np.allclose(calibration.bias_jaw, sim.params.bias_jaw_dps, atol=0.6)
