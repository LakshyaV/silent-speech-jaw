from __future__ import annotations

import numpy as np
import pytest

from ssj.data import FeatureDataset
from ssj.dsp import N_FEATURES, FrontendOutput, calibrate_session
from ssj.io import ReplaySource, Session
from ssj.runtime.live import FeatureRing, LiveDecoder, run_calibration
from ssj.runtime.pipeline import builtin_lm, load_lexicon, load_lm


# ------------------------------------------------------------------ resources
def test_language_resources(inventory, lexicon, tmp_path):
    assert len(load_lexicon("", inventory)) == len(lexicon)
    (tmp_path / "mini.dict").write_text("hello HH AH0 L OW1\nworld W ER1 L D\n")
    mini = load_lexicon(str(tmp_path / "mini.dict"), inventory)
    assert {"hello", "world", "<clench>"} == set(mini.words)
    lm = builtin_lm(mini)
    assert "hello" in lm and "world" in lm and lm.order == 2
    lm.to_arpa(tmp_path / "lm.arpa")
    assert load_lm(str(tmp_path / "lm.arpa"), mini).n_grams() == lm.n_grams()
    assert load_lm("", mini).n_grams() == lm.n_grams()  # the built-in fallback


def test_pipeline_decodes_features(pipeline, dataset_dir):
    data = FeatureDataset(dataset_dir)
    item = data.split("test")[0]
    result = pipeline.decode_features(data.load(item))
    assert "recogniser" in result.timings and isinstance(result.words, tuple)
    assert all(word in pipeline.lexicon for word in result.words)
    assert "LLM: none" in pipeline.describe() and "n-gram: order 3" in pipeline.describe()
    assert pipeline.decode_features(np.zeros((0, N_FEATURES), dtype=np.float32)).words == ()


# ----------------------------------------------------------------------- live
def test_feature_ring():
    ring = FeatureRing(capacity=50)
    for start in range(0, 200, 20):
        block = np.arange(start, start + 20, dtype=np.float32)[:, None].repeat(N_FEATURES, axis=1)
        ring.append(FrontendOutput(start, block, np.zeros(20, np.float32), np.zeros(20, bool), np.ones(20, bool)))
    assert ring.end == 200
    assert ring.slice(170, 190)[:, 0].tolist() == list(range(170, 190))
    assert ring.slice(0, 150)[:, 0].tolist() == list(range(140, 150))  # older frames were discarded
    assert len(ring.slice(500, 600)) == 0


def test_live_decoder_matches_offline_segments(pipeline, sim_root):
    """Replaying a session through the live loop finds the prompted utterances."""
    session = Session(sim_root / "sim" / "s02")
    calibration = calibrate_session(session, pipeline.bundle.frontend_config.fs_internal)
    seen = []
    decoder = LiveDecoder(ReplaySource(session, chunk_s=0.03), pipeline, calibration=calibration, on_event=seen.append)
    events = decoder.run()
    assert events == seen
    calibration_end = max(end for _, end in session.calibration_segments().values()) * 1e-6
    speech = [e for e in events if e.start_s > calibration_end]
    utterances = session.utterances()
    assert abs(len(speech) - len(utterances)) <= 2
    matched = 0
    for event in speech:
        middle = (event.start_s + event.duration_s / 2) * 1e6
        matched += any(u.t_start_us <= middle <= u.t_end_us for u in utterances)
        assert event.latency_s < 5.0 and 0.2 < event.duration_s < 13.0
    assert matched >= len(utterances) - 2
    assert all(isinstance(e.result.phonemes, list) for e in events)


def test_live_calibration_routine(pipeline, sim_root):
    session = Session(sim_root / "sim" / "s00")
    start = min(a for a, _ in session.calibration_segments().values())
    raw = session.raw()
    # The simulated wearer performs rest / head / jaw exactly when the routine asks for them.
    source = ReplaySource(raw[raw["t_us"] >= start - 800_000], session.info, chunk_s=0.02)
    messages: list[str] = []
    calibration = run_calibration(source, 200.0, say=messages.append, rest_s=3.0, head_s=6.0, jaw_s=5.5)
    assert [m.split()[1] for m in messages[:3]] == ["1/3", "2/3", "3/3"]
    assert calibration.level in ("full", "partial") and calibration.quality["source"] == "routine"
    assert np.linalg.norm(calibration.g_jaw) == pytest.approx(1.0, abs=0.05)
