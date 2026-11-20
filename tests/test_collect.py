from __future__ import annotations

import io

import numpy as np
import pytest

from ssj.collect import (
    ConsoleUI,
    Prompt,
    QCConfig,
    Recorder,
    RecorderConfig,
    build_prompt_set,
    builtin_sentences,
    check_stream,
    check_utterance,
    coverage_report,
    load_prompts,
    save_prompts,
)
from ssj.collect.prompts import diphones, greedy_cover, positional_units, read_text_pool, word_drills
from ssj.dsp import session_health
from ssj.io import DeviceInfo, ReplaySource, Session, SessionWriter
from ssj.io.protocol import SENSOR_JAW, SENSOR_REF, split_sensors
from ssj.phonology import normalize_text
from ssj.sim import ScriptItem, random_session_params, render_session, utterance_item
from ssj.sim.articulation import open_close, rest


# ------------------------------------------------------------------ prompts
def test_builtin_sentences_are_pronounceable(lexicon):
    sentences = builtin_sentences()
    assert len(sentences) >= 150 and len(set(sentences)) == len(sentences)
    for sentence in sentences:
        assert not lexicon.missing(normalize_text(sentence)), sentence


def test_units():
    assert diphones(["K", "AE", "T"]) == [("#", "K"), ("K", "AE"), ("AE", "T"), ("T", "#")]
    assert positional_units(["K", "AE", "T"]) == [("K", "initial"), ("AE", "medial"), ("T", "final")]


def test_greedy_cover_prefers_new_units_and_balances_counts():
    candidates = [["a", "b"], ["a", "b"], ["c"], ["a", "c", "d"], []]
    assert set(greedy_cover(candidates, 2)) == {0, 3} or set(greedy_cover(candidates, 2)) == {1, 3}
    everything = greedy_cover(candidates, 10)
    assert sorted(everything) == [0, 1, 2, 3]  # the empty candidate is never worth picking
    assert greedy_cover([], 3) == []


def test_word_drills_cover_every_phoneme_position(lexicon, inventory):
    words = word_drills(lexicon, inventory, 120)
    assert len(words) == len(set(words)) == 120
    seen = {unit for word in words for unit in positional_units(lexicon.prons(word)[0])}
    speech = [p for p, k in zip(inventory.phonemes, inventory.kinds) if k != "gesture"]
    assert {p for p, _ in seen} == set(speech)
    assert len(seen) > 95  # most (phoneme, position) pairs that exist in English words


def test_prompt_set_covers_the_inventory(lexicon, inventory, tmp_path):
    prompts = build_prompt_set(lexicon, inventory, n_words=40, n_sentences=60, seed=1, clench_every=10)
    report = coverage_report(prompts, inventory)
    assert report["prompts"] == 100 and report["missing_phonemes"] == []
    assert report["min_phoneme_count"] >= 3 and report["distinct_diphones"] > 400
    assert sum(p.text.endswith("<clench>") for p in prompts) == 6
    assert [p.kind for p in prompts[:40]] == ["word"] * 40
    other = build_prompt_set(lexicon, inventory, n_words=40, n_sentences=60, seed=2)
    assert {p.text for p in other} != {p.text for p in prompts}  # another seed, another session's list
    save_prompts(tmp_path / "p.jsonl", prompts)
    assert [p.to_dict() for p in load_prompts(tmp_path / "p.jsonl")] == [p.to_dict() for p in prompts]


def test_text_pool_reader_handles_plain_and_tsv(tmp_path):
    (tmp_path / "pool.tsv").write_text("# header\n12\teng\tHello there.\n\n13\teng\tHow are you?\n")
    assert read_text_pool(tmp_path / "pool.tsv") == ["Hello there.", "How are you?"]
    assert read_text_pool(tmp_path / "pool.tsv", limit=1) == ["Hello there."]


# ----------------------------------------------------------------------- QC
def test_stream_check_flags_transport_problems(sim_session):
    raw, info = sim_session.raw[:4000], sim_session.info
    assert check_stream(raw, info).ok
    thinned = raw[::3]  # two thirds of the frames lost
    result = check_stream(thinned, info)
    assert not result.ok and {"low_rate", "frame_loss"} <= set(result.problems)
    assert "missing_sensor" in check_stream(raw[raw["sensor"] == 0], info).problems
    saturated = raw.copy()
    saturated["gyro"][10] = 32767
    assert "saturated" in check_stream(saturated, info).problems
    assert check_stream(thinned, info).advice()


def test_stream_health_counts_delivered_samples(sim_session):
    raw, info = sim_session.raw[:6000], sim_session.info
    jaw = split_sensors(raw)[SENSOR_JAW]
    # FIFO-mode timestamps sit on the board's drain grid: the typical interval says
    # 250 Hz, the number of samples per second says 208.
    assert np.median(np.diff(jaw["t_us"])) == 4000
    assert session_health(jaw["t_us"], jaw["seq"], info.odr_hz)["rate_hz"] == pytest.approx(info.odr_hz, rel=0.02)

    # A board that falls behind loses batches to FIFO overruns and numbers what it
    # does send consecutively: nominal spacing inside every batch, no sequence gap,
    # and a third of the signal missing.
    period_us = 1e6 / info.odr_hz
    parts = []
    for sensor in (SENSOR_JAW, SENSOR_REF):
        part = raw[raw["sensor"] == sensor].copy()
        index = np.arange(len(part))
        part["t_us"] = part["t_us"][0] + np.round(index * period_us).astype(np.int64)
        part = part[index % 12 < 8]
        part["seq"] = np.arange(len(part))
        parts.append(part)
    starved = np.concatenate(parts)
    starved = starved[np.argsort(starved["t_us"], kind="stable")]
    result = check_stream(starved, info)
    assert result.problems == ["low_rate"]
    assert result.metrics["jaw_rate_hz"] == pytest.approx(info.odr_hz * 2 / 3, rel=0.02)


def test_utterance_check_flags_timing_problems(sim_session):
    raw, info = sim_session.raw[:3000], sim_session.info
    quiet = np.zeros(300, dtype=np.float32)
    assert check_utterance(raw, info, quiet, 100.0).problems == ["no_activity"]
    good = quiet.copy()
    good[80:200] = 8.0
    assert check_utterance(raw, info, good, 100.0).ok
    early = quiet.copy()
    early[0:120] = 8.0
    assert "started_early" in check_utterance(raw, info, early, 100.0).problems
    late = quiet.copy()
    late[200:300] = 8.0
    assert "cut_off" in check_utterance(raw, info, late, 100.0).problems
    result = check_utterance(raw, info, good, 100.0, head_rate_dps=90.0, config=QCConfig(max_head_rate_dps=60.0))
    assert result.problems == ["head_motion"] and result.to_dict()["metrics"]["active_s"] == pytest.approx(1.2, abs=0.1)


# ----------------------------------------------------------------- recorder
class ScriptedWearer:
    """Plays the part of the person and the hardware for a recorder rehearsal.

    The recorder is driven by device time, so a pre-rendered stream in which the
    wearer does the right thing at the right time exercises the whole protocol.
    """

    def __init__(
        self,
        lexicon,
        prompts: list[Prompt],
        config: RecorderConfig,
        seed: int = 0,
        silent: frozenset[int] = frozenset(),
    ):
        rng = np.random.default_rng(seed)
        script = [
            ScriptItem("gap", rest(2.0 + 0.8)),  # link check + settle
            ScriptItem("rest", rest(config.calib_rest_s + 0.8)),
            ScriptItem("head", rest(config.calib_head_s + 0.8), head_motion=True),
            ScriptItem("jaw", open_close(5, rng, period_s=config.calib_jaw_s / 5)),
            ScriptItem("gap", rest(0.6 + 1.2 + 0.4)),  # tail of the routine + tracker warm-up
        ]
        for index, prompt in enumerate(prompts):
            window = config.window_s(len(prompt.phonemes))
            item = utterance_item(prompt.text, lexicon, rng, rate=1.2)
            spoken = item.trajectory.duration_s
            lead = config.pre_rest_s + 0.35  # reaction time after GO
            trailing = max(window - 0.35 - spoken, 0.05) + max(config.post_rest_s, config.tail_margin_s + 0.05)
            script.append(ScriptItem("gap", rest(lead)))
            script.append(ScriptItem("gap", rest(spoken)) if index in silent else ScriptItem("gap", item.trajectory))
            script.append(ScriptItem("gap", rest(trailing)))
        script.append(ScriptItem("gap", rest(2.0)))
        self.session = render_session(script, random_session_params(rng), rng, gap_s=(0.0, 0.0), t0_s=1.0)


def test_recorder_runs_a_full_session(lexicon, tmp_path):
    config = RecorderConfig(auto=True, max_retries=0)
    prompts = [
        Prompt(f"p{i}", text, "sentence", lexicon.phonemize(text))
        for i, text in enumerate(["hello how are you", "thank you very much", "what time is it"])
    ]
    wearer = ScriptedWearer(lexicon, prompts, config, silent=frozenset({1}))
    source = ReplaySource(wearer.session.raw, wearer.session.info, chunk_s=0.02)
    out = io.StringIO()
    with SessionWriter(tmp_path / "s01" / "a", subject="s01", session="a", info=wearer.session.info) as writer:
        recorder = Recorder(source, writer, config=config, ui=ConsoleUI(out=out))
        counts = recorder.run(prompts)
        writer.close(calibration=recorder.calibration.to_dict())
    text = out.getvalue()
    assert "CALIBRATION" in text and "GO" in text and "SESSION COMPLETE" in text
    assert recorder.calibration.level == "full"
    assert counts == {"ok": 2, "bad": 1, "skipped": 0}

    session = Session(tmp_path / "s01" / "a")
    assert set(session.calibration_segments()) == {"rest", "head", "jaw"}
    recorded = session.utterances(include_bad=True)
    assert [u.text for u in recorded] == [p.text for p in prompts]
    assert [u.status for u in recorded] == ["ok", "bad", "ok"]
    assert recorded[1].qc["problems"] == ["no_activity"]  # the wearer stayed silent on the second prompt
    assert recorded[0].phonemes == prompts[0].phonemes and recorded[0].prompt_id == "p0"
    for utterance in recorded:
        assert 1.5 < (utterance.t_end_us - utterance.t_start_us) * 1e-6 < 6.0
    assert len(session.raw()) > 0.9 * len(wearer.session.raw[wearer.session.raw["t_us"] <= session.raw()["t_us"].max()])
    assert session.meta["calibration"]["quality"]["level"] == "full"


def test_recorder_interactive_choices(lexicon, tmp_path):
    config = RecorderConfig(max_retries=1)
    prompts = [
        Prompt(f"p{i}", text, "sentence", lexicon.phonemize(text))
        for i, text in enumerate(["good morning everyone", "see you later"])
    ]
    # prompt 0 is taken twice (the wearer asks for a redo), prompt 1 is skipped
    doubled = [prompts[0], prompts[0], prompts[1]]
    wearer = ScriptedWearer(lexicon, doubled, config)
    answers = iter(["r", "", "s"])
    source = ReplaySource(wearer.session.raw, wearer.session.info, chunk_s=0.02)
    with SessionWriter(tmp_path / "x", subject="s", session="x", info=wearer.session.info) as writer:
        recorder = Recorder(source, writer, config=config, ui=ConsoleUI(ask=lambda _: next(answers), out=io.StringIO()))
        counts = recorder.run(prompts)
    statuses = [(u.text, u.status) for u in Session(tmp_path / "x").utterances(include_bad=True)]
    assert statuses == [("good morning everyone", "bad"), ("good morning everyone", "ok"), ("see you later", "bad")]
    assert counts["ok"] == 1 and counts["skipped"] == 1


def test_recorder_stops_cleanly_when_the_stream_ends(lexicon, tmp_path):
    info = DeviceInfo()
    wearer = ScriptedWearer(lexicon, [], RecorderConfig())
    short = wearer.session.raw[:3000]
    with SessionWriter(tmp_path / "y", subject="s", session="y", info=info) as writer:
        counts = Recorder(ReplaySource(short, info), writer, ui=ConsoleUI(out=io.StringIO())).run(
            [Prompt("p", "hello", "word", ["HH", "AH", "L", "OW"])]
        )
    assert counts == {"ok": 0, "bad": 0, "skipped": 0}
    assert len(Session(tmp_path / "y").raw()) == len(short)
