from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ssj.dsp import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    N_FEATURES,
    Calibration,
    Frontend,
    FrontendConfig,
    FrontendOutput,
    Segmenter,
    SegmenterConfig,
    calibrate_session,
    frames_for_interval,
    process_session,
    segment_activity,
    trim_to_activity,
)
from ssj.io import Session


@pytest.fixture(scope="module")
def processed(sim_session, tmp_path_factory):
    directory = tmp_path_factory.mktemp("frontend") / "s"
    sim_session.write(directory, subject="sim", session="s")
    session = Session(directory)
    calibration = calibrate_session(session)
    return session, calibration, process_session(session, FrontendConfig(), calibration)


def test_feature_layout():
    assert N_FEATURES == 23 == len(FEATURE_NAMES)
    covered = sorted(i for a, b in FEATURE_GROUPS.values() for i in range(a, b))
    assert covered == list(range(N_FEATURES))  # the groups tile the feature vector exactly


def test_fingerprint_tracks_the_configuration():
    assert FrontendConfig().fingerprint == FrontendConfig().fingerprint
    assert FrontendConfig().fingerprint != replace(FrontendConfig(), lowpass_hz=25.0).fingerprint


def test_streaming_equals_batch(sim_session, processed):
    """Live chunks of any size must give bit-identical features to offline processing."""
    _, calibration, reference = processed
    rng = np.random.default_rng(0)
    raw = sim_session.raw[: 2 * 208 * 25]  # 25 s is plenty to cross every state
    one_shot = Frontend(FrontendConfig(), calibration).push(raw, sim_session.info)
    frontend = Frontend(FrontendConfig(), calibration)
    parts, cursor = [], 0
    while cursor < len(raw):
        step = int(rng.integers(1, 60))
        parts.append(frontend.push(raw[cursor : cursor + step], sim_session.info))
        cursor += step
    chunked = FrontendOutput.concatenate(parts)
    assert chunked.k0 == one_shot.k0 and len(chunked) == len(one_shot)
    assert np.array_equal(chunked.feats, one_shot.feats)
    assert np.array_equal(chunked.activity, one_shot.activity)
    assert np.array_equal(chunked.rest, one_shot.rest) and np.array_equal(chunked.valid, one_shot.valid)
    assert np.array_equal(one_shot.feats, reference.feats[: len(one_shot)])


def test_relative_stream_tracks_the_jaw(sim_session, processed):
    """Channel 12 is opening velocity and channel 18 opening angle, whatever the mounting."""
    _, _, out = processed
    cfg = FrontendConfig()
    t_true = 5.0 + np.arange(len(sim_session.opening_deg)) / sim_session.fs
    t_feat = (out.k0 + np.arange(len(out))) / cfg.fs
    velocity = np.interp(t_feat, t_true, np.gradient(sim_session.opening_deg) * sim_session.fs)
    angle = np.interp(t_feat, t_true, sim_session.opening_deg)
    lag = 2  # the anti-alias filter delays the features by about two frames
    assert np.corrcoef(out.feats[lag:, 12], velocity[:-lag])[0, 1] > 0.97
    assert np.corrcoef(out.feats[lag:, 18], angle[:-lag])[0, 1] > 0.97
    assert np.std(out.feats[:, 12] * cfg.gyro_scale_dps - np.roll(velocity, lag))[...] < 0.3 * np.std(velocity)


def test_head_motion_cancels_in_the_relative_stream(sim_session, processed):
    """During the head-only calibration phase the jaw and reference streams move, rel does not."""
    _, _, out = processed
    cfg = FrontendConfig()
    head = next(e for e in sim_session.events if e.get("phase") == "head")
    a, b = frames_for_interval(out, head["t_start_us"], head["t_end_us"], cfg.fs)
    jaw_rate = np.sqrt(np.mean(out.feats[a:b, 0:3] ** 2))
    ref_rate = np.sqrt(np.mean(out.feats[a:b, 6:9] ** 2))
    rel_rate = np.sqrt(np.mean(out.feats[a:b, 12:15] ** 2))
    assert jaw_rate > 0.1 and ref_rate > 0.1  # tens of deg/s of head rotation
    assert rel_rate < 0.08 * jaw_rate
    rel_accel = np.sqrt(np.mean(out.feats[a:b, 15:18] ** 2))
    jaw_accel = np.sqrt(np.mean(out.feats[a:b, 3:6] ** 2))
    # The gravity shift is common to both sensors and cancels; what remains is the
    # lever-arm acceleration, which differs between two points on a turning head.
    assert rel_accel < 0.5 * jaw_accel


def test_without_calibration_the_streams_do_not_cancel(sim_session):
    out = Frontend(FrontendConfig(), Calibration.identity()).push(sim_session.raw, sim_session.info)
    head = next(e for e in sim_session.events if e.get("phase") == "head")
    a, b = frames_for_interval(out, head["t_start_us"], head["t_end_us"], 100.0)
    assert np.sqrt(np.mean(out.feats[a:b, 12:15] ** 2)) > 0.3 * np.sqrt(np.mean(out.feats[a:b, 0:3] ** 2))


def test_rest_tracking_and_activity(sim_session, processed):
    _, _, out = processed
    cfg = FrontendConfig()
    rest_event = next(e for e in sim_session.events if e.get("phase") == "rest")
    a, b = frames_for_interval(out, rest_event["t_start_us"], rest_event["t_end_us"], cfg.fs)
    assert out.rest[a + 110 : b].mean() > 0.9 and out.activity[a + 110 : b].max() < 3.0
    for utterance in sim_session.utterances:
        a, b = frames_for_interval(out, utterance.t_start_us, utterance.t_end_us, cfg.fs)
        assert out.activity[a:b].max() > 6.0  # articulation is far above the onset threshold
        assert out.rest[a:b].mean() < 0.8
    assert out.valid.mean() > 0.999 and np.isfinite(out.feats).all()


def test_segmenter_finds_every_utterance(sim_session, processed):
    _, _, out = processed
    fs = FrontendConfig().fs
    segments = segment_activity(out.activity, SegmenterConfig(), fs, out.k0)
    calibration_end = max(e["t_end_us"] for e in sim_session.events if e["type"] == "calibration") * 1e-6
    speech = [s for s in segments if s.start / fs > calibration_end]
    assert len(speech) == len(sim_session.utterances)
    for segment, utterance in zip(speech, sim_session.utterances):
        assert utterance.t_start_us * 1e-6 - 0.3 <= segment.start / fs <= utterance.t_end_us * 1e-6
        assert segment.end / fs <= utterance.t_end_us * 1e-6 + 0.5
        assert 0.3 < len(segment) / fs < 6.0


def test_segmenter_state_machine():
    fs = 100.0
    cfg = SegmenterConfig(
        start_frames=3,
        end_quiet_s=0.5,
        pre_roll_s=0.1,
        post_roll_s=0.1,
        min_duration_s=0.2,
        max_duration_s=3.0,
        cooldown_s=0.0,
    )
    score = np.zeros(1500, dtype=np.float32)
    score[100:104] = 5.0  # a twitch: too short to be speech
    score[300:380] = 5.0  # an utterance ...
    score[400:460] = 5.0  # ... with a word gap shorter than end_quiet_s
    score[700:1200] = 5.0  # longer than max_duration_s: must be cut
    segments = segment_activity(score, cfg, fs)
    assert [(s.start, s.end, s.forced) for s in segments[:1]] == [(290, 470, False)]
    assert segments[1].forced and len(segments[1]) <= 3.0 * fs + 20
    streaming = Segmenter(cfg, fs)
    collected = []
    for start in range(0, 1500, 37):
        collected += streaming.push(start, score[start : start + 37])
    collected += streaming.flush(1500)
    assert [(s.start, s.end) for s in collected] == [(s.start, s.end) for s in segments]


def test_trim_to_activity():
    score = np.zeros(400, dtype=np.float32)
    score[120:250] = 6.0
    a, b = trim_to_activity(score, SegmenterConfig(pre_roll_s=0.2, post_roll_s=0.1), 100.0)
    assert (a, b) == (100, 260)
    assert trim_to_activity(np.zeros(50, dtype=np.float32)) == (0, 50)  # nothing detected: keep the window
