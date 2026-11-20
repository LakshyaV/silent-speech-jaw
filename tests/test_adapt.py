from __future__ import annotations

import json

import pytest
import torch

from ssj.cli import main
from ssj.train.adapt import AdaptConfig, adapt_session
from ssj.train.trainer import load_checkpoint


def test_adapt_command(dataset_dir, run_dir, tmp_path):
    adapted = tmp_path / "adapted"
    arguments = [
        "adapt",
        "--checkpoint",
        str(run_dir / "checkpoint.pt"),
        "--data",
        str(dataset_dir),
        "--session",
        "sim/s02",
    ]
    assert main([*arguments, "--out", str(adapted), "-s", "adapt.epochs=2"]) == 0
    summary = json.loads((adapted / "adaptation.json").read_text())
    assert summary["session"] == "sim/s02" and summary["after"]["per"] <= summary["before"]["per"] + 1e-9
    meta = json.loads((adapted / "bundle" / "bundle.json").read_text())
    assert meta["baked_session"] == "sim/s02" and "sim/s02" in meta["sessions"]


def test_adaptation_only_moves_the_new_adapter(dataset_dir, run_dir, tmp_path):
    out = tmp_path / "adapted.pt"
    adapt_session(run_dir / "checkpoint.pt", dataset_dir, "sim/s02", out, AdaptConfig(epochs=2), report=lambda m: None)
    before, payload_before = load_checkpoint(run_dir / "checkpoint.pt")
    after, payload_after = load_checkpoint(out)
    assert payload_after["sessions"] == [*payload_before["sessions"], "sim/s02"]
    assert after.adapter_delta.shape[0] == before.adapter_delta.shape[0] + 1
    for (name, old), (_, new) in zip(before.state_dict().items(), after.state_dict().items()):
        if name.startswith("adapter_"):
            assert torch.equal(old, new[: old.shape[0]]), name  # existing adapters are untouched
        else:
            assert torch.equal(old, new), name  # the recogniser is frozen
    with pytest.raises(ValueError):
        adapt_session(run_dir / "checkpoint.pt", dataset_dir, "sim/nope", out)
