"""Command-line workflow tests: the commands are thin, so one pass through each is enough."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ssj import assets
from ssj.cli import main
from ssj.io import Session, find_sessions

ROOT = Path(__file__).resolve().parents[1]


def test_prompts_command(tmp_path, capsys):
    out = tmp_path / "prompts.jsonl"
    assert (
        main(
            [
                "prompts",
                "build",
                "--out",
                str(out),
                "--words",
                "10",
                "--sentences",
                "15",
                "--seed",
                "4",
                "--clench-every",
                "5",
            ]
        )
        == 0
    )
    assert "39/39 phonemes" in capsys.readouterr().out
    assert len(out.read_text().splitlines()) == 25
    assert main(["prompts", "show", str(out)]) == 0
    assert "distinct diphones" in capsys.readouterr().out


def test_sim_data_and_lm_commands(tmp_path, capsys):
    raw = tmp_path / "raw"
    assert (
        main(["sim", "--out", str(raw), "--sessions", "2", "--utterances", "6", "--seed", "5", "--mode", "polled"]) == 0
    )
    sessions = list(find_sessions(raw))
    assert [s.key for s in sessions] == ["sim/s00", "sim/s01"] and len(sessions[0].utterances()) == 6
    assert sessions[0].info.mode_name == "polled"

    assert main(["data", "build", "--raw", str(raw), "--out", str(tmp_path / "data"), "-s", "dataset.trim=false"]) == 0
    assert main(["data", "info", str(tmp_path / "data")]) == 0
    text = capsys.readouterr().out
    assert "front-end fingerprint" in text and "sim/s01" in text

    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "\n".join(["hello how are you", "i need some water", "thank you very much", "how are you today"] * 5)
    )
    lm = tmp_path / "lm.npz"
    assert (
        main(
            [
                "lm",
                "build",
                "--text",
                str(corpus),
                "--out",
                str(lm),
                "--order",
                "2",
                "--min-count",
                "1",
                "--dir",
                str(tmp_path / "none"),
                "--arpa",
                str(tmp_path / "lm.arpa"),
            ]
        )
        == 0
    )
    assert main(["lm", "info", str(lm), "--sentence", "how are you"]) == 0
    assert "log P(how are you)" in capsys.readouterr().out
    assert (tmp_path / "lm.arpa").read_text().startswith("\\data\\")


def test_record_and_monitor_against_a_replayed_session(sim_root, tmp_path, capsys):
    source = f"replay:{sim_root / 'sim' / 's00'}"
    assert main(["monitor", "--source", source, "--seconds", "0.3", "--interval", "0.05"]) == 0
    assert "fifo mode" in capsys.readouterr().out
    # A replayed stream has no wearer following the cues; this exercises the plumbing only.
    assert (
        main(
            [
                "record",
                "--subject",
                "t01",
                "--session",
                "dry",
                "--source",
                source,
                "--out",
                str(tmp_path),
                "--words",
                "1",
                "--sentences",
                "1",
                "--auto",
                "--note",
                "dry run",
            ]
        )
        == 0
    )
    session = Session(tmp_path / "t01" / "dry")
    assert session.meta["note"] == "dry run" and set(session.calibration_segments()) == {"rest", "head", "jaw"}
    assert len(session.raw()) > 1000


def test_train_and_export_commands(dataset_dir, run_dir, tmp_path, capsys):
    small = [
        "-s",
        "model.stem_channels=8",
        "-s",
        "model.d_model=32",
        "-s",
        "model.tcn_dilations=[1]",
        "-s",
        "model.gru_hidden=16",
        "-s",
        "model.gru_layers=1",
    ]
    run = tmp_path / "run"
    assert (
        main(
            [
                "train",
                "--data",
                str(dataset_dir),
                "--out",
                str(run),
                "--init",
                str(run_dir / "checkpoint.pt"),
                *small,
                "-s",
                "train.epochs=1",
                "-s",
                "train.batch_size=12",
                "-s",
                "train.threads=1",
            ]
        )
        == 0
    )
    assert (run / "bundle" / "model.onnx").exists() and "bundle ->" in capsys.readouterr().out
    assert main(["export", "--checkpoint", str(run / "checkpoint.pt"), "--out", str(tmp_path / "b")]) == 0
    assert "max deviation from checkpoint" in capsys.readouterr().out


def test_assets(tmp_path, monkeypatch, capsys):
    root = assets.assets_dir(tmp_path)
    assert assets.find_llm(root) is None
    (root / "llm").mkdir()
    (root / "llm" / "a.gguf").write_bytes(b"x")
    assert assets.find_llm(root) == root / "llm" / "a.gguf"
    (root / "llm" / "b.gguf").write_bytes(b"x")
    assert assets.find_llm(root) is None  # ambiguous: the user must choose
    monkeypatch.setenv("SSJ_LLM", "/models/x.gguf")
    assert assets.find_llm(root) == Path("/models/x.gguf")
    monkeypatch.setenv("SSJ_ASSETS", str(tmp_path / "elsewhere"))
    assert assets.assets_dir() == tmp_path / "elsewhere"
    assert main(["assets", "list", "--dir", str(root)]) == 0
    listing = capsys.readouterr().out
    assert "missing" in listing and all(name in listing for name in assets.LLM_PRESETS)
    for preset in assets.LLM_PRESETS.values():
        assert preset.url.startswith("https://huggingface.co/") and preset.url.endswith(".gguf")

    served = tmp_path / "served.bin"
    served.write_bytes(np.random.default_rng(0).bytes(300_000))
    target = tmp_path / "dl" / "copy.bin"
    assert assets.download(served.as_uri(), target).read_bytes() == served.read_bytes()
    assert assets.download("file:///does/not/exist", target) == target  # already present: no request made
