from __future__ import annotations

import numpy as np
import pytest

from ssj.io import RAW_DTYPE, Session, split_sensors
from ssj.io.protocol import MODE_FIFO, MODE_POLLED, SENSOR_JAW, SENSOR_REF
from ssj.sim import PHONE_SPECS, simulate_session, synthesize
from ssj.sim.imu import random_rotation, random_session_params


def test_every_speech_phoneme_has_a_spec(inventory):
    speech = [p for p, k in zip(inventory.phonemes, inventory.kinds) if k != "gesture"]
    assert set(speech) == set(PHONE_SPECS)


def test_voicing_pairs_are_indistinguishable_by_construction():
    """The simulator must not leak information a jaw sensor cannot have."""
    for a, b in [("P", "B"), ("T", "D"), ("K", "G"), ("F", "V"), ("S", "Z"), ("SH", "ZH"), ("CH", "JH"), ("TH", "DH")]:
        assert PHONE_SPECS[a] == PHONE_SPECS[b]
        one = synthesize(["AA", a, "IY"], np.random.default_rng(7))
        two = synthesize(["AA", b, "IY"], np.random.default_rng(7))
        assert np.array_equal(one.opening_deg, two.opening_deg) and np.array_equal(one.tissue_deg, two.tissue_deg)
    wide = synthesize(["AA"], np.random.default_rng(1), rate=1.0, effort=1.0)
    narrow = synthesize(["IY"], np.random.default_rng(1), rate=1.0, effort=1.0)
    assert wide.opening_deg.max() > 2.5 * narrow.opening_deg.max()  # vowel height is visible


def test_trajectory_is_smooth_and_returns_to_rest():
    trajectory = synthesize("HH AH L OW W ER L D".split(), np.random.default_rng(0), word_boundaries=[0, 4])
    assert trajectory.duration_s > 0.8
    assert abs(trajectory.opening_deg[0]) < 0.05 and abs(trajectory.opening_deg[-1]) < 0.3
    velocity = np.gradient(trajectory.opening_deg) * trajectory.fs
    acceleration = np.gradient(velocity) * trajectory.fs
    assert np.abs(velocity).max() < 400  # deg/s
    assert np.abs(np.radians(acceleration)).max() * 0.06 / 9.81 < 1.5  # g at a 6 cm lever arm
    assert [p for p, _, _ in trajectory.phone_times] == "HH AH L OW W ER L D".split()
    assert all(b > a for _, a, b in trajectory.phone_times)


def test_clench_is_a_vibration_burst_without_opening():
    trajectory = synthesize(["CLENCH"], np.random.default_rng(0))
    assert np.abs(trajectory.vibration_g).max() > 0.01
    assert trajectory.opening_deg.max() < 0.2


def test_session_is_deterministic_and_well_formed(lexicon):
    texts = ["hello how are you today", "water"]
    one = simulate_session(texts, lexicon, seed=11)
    two = simulate_session(texts, lexicon, seed=11)
    other = simulate_session(texts, lexicon, seed=12)
    assert np.array_equal(one.raw, two.raw)
    assert not np.array_equal(one.raw[:1000], other.raw[:1000])
    assert one.raw.dtype == RAW_DTYPE
    parts = split_sensors(one.raw)
    for sensor in (SENSOR_JAW, SENSOR_REF):
        part = parts[sensor]
        assert np.all(np.diff(part["seq"].astype(np.int64)) == 1)
        assert np.all(np.diff(part["t_us"]) >= 0)
        rate = (len(part) - 1) / ((part["t_us"][-1] - part["t_us"][0]) * 1e-6)
        assert 200 < rate < 216  # nominal 208 Hz, each sensor with its own clock error
    # At rest each accelerometer reads 1 g in whatever direction its mount points.
    rest = parts[SENSOR_JAW][:200]["accel"].astype(float) * one.info.accel_g_per_lsb
    assert np.linalg.norm(rest.mean(axis=0)) == pytest.approx(1.0, abs=0.03)
    assert [u.text for u in one.utterances] == texts
    assert one.utterances[0].phonemes[:4] == ["HH", "AH", "L", "OW"] or one.utterances[0].phonemes[:4] == [
        "HH",
        "EH",
        "L",
        "OW",
    ]
    phases = [e["phase"] for e in one.events if e["type"] == "calibration"]
    assert phases == ["rest", "head", "jaw"]


def test_polled_mode_has_jitter_and_fifo_mode_has_stamped_batches(lexicon):
    fifo = simulate_session(["hello"], lexicon, seed=1, mode=MODE_FIFO)
    polled = simulate_session(["hello"], lexicon, seed=1, mode=MODE_POLLED)
    dt_fifo = np.diff(split_sensors(fifo.raw)[SENSOR_JAW]["t_us"])
    dt_polled = np.diff(split_sensors(polled.raw)[SENSOR_JAW]["t_us"])
    assert set(np.unique(dt_fifo)) <= {4000, 8000}  # quantised to the MCU's drain period
    assert len(np.unique(dt_polled)) > 50 and dt_polled.max() > 1.5 * np.median(dt_polled)


def test_sessions_differ_in_mounting_and_write_to_disk(tmp_path, lexicon):
    rng = np.random.default_rng(0)
    a, b = random_session_params(rng), random_session_params(rng)
    assert not np.allclose(a.R_mount_jaw, b.R_mount_jaw)
    for R in (a.R_mount_jaw, a.R_mount_ref, random_rotation(rng, 20.0)):
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.linalg.det(R) == pytest.approx(1.0)
    cone = [np.degrees(np.arccos((np.trace(random_rotation(rng, 20.0)) - 1) / 2)) for _ in range(50)]
    assert max(cone) <= 20.0 + 1e-6

    sim = simulate_session(["thank you very much"], lexicon, seed=2, clench_rate=1.0)
    directory = sim.write(tmp_path / "sim" / "s00", subject="sim", session="s00")
    session = Session(directory)
    assert session.meta["simulated"] is True and "R_canon" in session.meta["truth"]
    assert np.array_equal(session.raw(), sim.raw)
    assert session.utterances()[0].text.endswith("<clench>")
    assert session.utterances()[0].phonemes[-1] == "CLENCH"
