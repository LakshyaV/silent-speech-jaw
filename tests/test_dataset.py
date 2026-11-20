from __future__ import annotations

import numpy as np
import pytest

from ssj.data import AugmentConfig, DatasetConfig, FeatureDataset, assign_splits, augment, build_dataset
from ssj.data.augment import inject_head_motion, time_stretch
from ssj.data.build import normalization_stats
from ssj.dsp import FEATURE_GROUPS, N_FEATURES, FrontendConfig


def test_split_rules():
    """Held-out sessions whenever possible; within-session splits are flagged."""
    three, notes = assign_splits(["a/1", "a/2", "a/3", "b/1", "b/2", "c/1"], DatasetConfig())
    assert three == {"a/1": "train", "a/2": "val", "a/3": "test", "b/1": "mixed-val", "b/2": "test", "c/1": "mixed"}
    assert any("do not measure robustness to re-mounting" in note for note in notes)
    explicit, _ = assign_splits(["a/1", "a/2", "a/3"], DatasetConfig(test_sessions=("a/1",), val_sessions=("a/2",)))
    assert explicit == {"a/1": "test", "a/2": "val", "a/3": "train"}
    everything, _ = assign_splits(["a/1", "a/2"], DatasetConfig(holdout="none"))
    assert set(everything.values()) == {"train"}
    with pytest.raises(ValueError):
        assign_splits(["a/1"], DatasetConfig(test_sessions=("nope/1",)))


def test_build_dataset(dataset_dir, sim_root, inventory):
    data = FeatureDataset(dataset_dir)
    meta = data.meta
    assert meta["frontend_fingerprint"] == FrontendConfig().fingerprint == data.frontend_config.fingerprint
    assert data.inventory == inventory
    assert data.sessions == ["sim/s00", "sim/s01", "sim/s02"]
    assert {key: value["split"] for key, value in meta["sessions"].items()} == {
        "sim/s00": "train",
        "sim/s01": "val",
        "sim/s02": "test",
    }
    assert all(value["calibration"]["quality"]["level"] == "full" for value in meta["sessions"].values())
    assert meta["splits"] == {"train": 26, "val": 26, "test": 26} and not meta["dropped"]
    assert not meta["split_notes"]

    item = data.split("train")[0]
    feats = data.load(item)
    assert feats.shape == (item.n_frames, N_FEATURES) and feats.dtype == np.float32
    assert all(p in inventory for p in item.phonemes)
    assert item.n_frames >= 4 * len(item.phonemes)
    # Trimming removes most of the cue window's dead time.
    lengths = {i.text: i.n_frames for i in data.items}
    assert lengths["water"] < 110 and 150 < lengths["the weather is beautiful today"] < 500
    assert any(i.text.endswith("<clench>") and i.phonemes[-1] == "CLENCH" for i in data.items)

    normalised = np.concatenate([data.normalize(data.load(i)) for i in data.split("train")])
    for start in (0, 3, 6, 9, 12, 15, 18):  # each 3-vector has unit RMS as a whole
        assert np.sqrt(np.mean(normalised[:, start : start + 3] ** 2)) == pytest.approx(1.0, abs=0.05)
    assert np.sqrt(np.mean(normalised[:, 12] ** 2)) > 3 * np.sqrt(
        np.mean(normalised[:, 13] ** 2)
    )  # hinge axis dominates


def test_single_session_dataset_is_flagged(tmp_path, sim_root, lexicon, inventory):
    meta = build_dataset(sim_root / "sim" / "s00", tmp_path / "one", lexicon, inventory)
    assert meta["sessions"]["sim/s00"]["split"] == "mixed"
    assert meta["splits"]["train"] > meta["splits"]["val"] > 0 and meta["splits"]["test"] > 0
    assert meta["split_notes"] and "single session" in meta["split_notes"][0]
    with pytest.raises(FileNotFoundError):
        build_dataset(tmp_path / "empty", tmp_path / "out", lexicon, inventory)


def test_normalization_keeps_vector_geometry():
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(500, N_FEATURES)).astype(np.float32)
    feats[:, 12] *= 20.0  # a dominant axis must stay dominant after scaling
    center, scale = normalization_stats(feats)
    assert scale[12] == scale[13] == scale[14]
    assert np.all(center[:21] == 0) and scale[21] != scale[22]


def test_time_stretch_scales_rates():
    feats = np.ones((100, N_FEATURES), dtype=np.float32)
    slow = time_stretch(feats, 1.25)
    assert len(slow) == 125
    assert np.allclose(slow[:, 0:3], 0.8) and np.allclose(slow[:, 12:15], 0.8)  # same movement, lower rate
    assert np.allclose(slow[:, 18:21], 1.0) and np.allclose(slow[:, 3:6], 1.0)  # angles and tilt are unchanged


def test_head_motion_injection_is_common_mode():
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(300, N_FEATURES)).astype(np.float32)
    out = inject_head_motion(feats, rng, 20.0, FrontendConfig())
    a, b = FEATURE_GROUPS["rel"]
    assert np.array_equal(out[:, a:b], feats[:, a:b])  # the relative stream carries no head motion
    assert np.allclose(
        out[:, 0:6] - feats[:, 0:6], out[:, 6:12] - feats[:, 6:12], atol=1e-5
    )  # identical on both sensors
    assert np.sqrt(np.mean((out[:, 0:3] - feats[:, 0:3]) ** 2)) * 50.0 == pytest.approx(20.0, rel=0.15)


def test_augment_is_well_behaved(dataset_dir):
    data = FeatureDataset(dataset_dir)
    feats = data.load(data.items[0])
    rng = np.random.default_rng(0)
    assert augment(feats, rng, AugmentConfig(enabled=False)) is feats
    lengths, similarity = set(), []
    for _ in range(30):
        out = augment(feats, rng, AugmentConfig())
        assert out.shape[1] == N_FEATURES and out.dtype == np.float32 and np.isfinite(out).all()
        assert 0.8 * len(feats) <= len(out) <= 1.25 * len(feats)
        lengths.add(len(out))
        stretched = np.interp(np.linspace(0, len(feats) - 1, len(out)), np.arange(len(feats)), feats[:, 18])
        similarity.append(np.corrcoef(out[:, 18], stretched)[0, 1])
    assert len(lengths) > 5
    # Augmentation perturbs, it does not destroy: the opening-angle channel stays recognisable.
    assert np.median(similarity) > 0.9 and min(similarity) > 0.5
