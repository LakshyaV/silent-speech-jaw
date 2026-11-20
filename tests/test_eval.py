from __future__ import annotations

import json

import numpy as np
import pytest

from ssj.cli import main
from ssj.data import FeatureDataset
from ssj.decode.beam import BeamConfig
from ssj.eval.confusion import confusion, describe_classes, refine_classes, similarity
from ssj.eval.report import UtteranceRecord, evaluate, format_metrics, phoneme_sequences, sample_lines
from ssj.eval.tune import as_overrides, tune_first_pass, tune_fusion
from ssj.llm import FusionConfig
from ssj.llm.fusion import Candidate
from ssj.phonology import Inventory
from ssj.runtime.bundle import Bundle


# ----------------------------------------------------------------- evaluation
def test_evaluate_reports_every_stage(pipeline, dataset_dir):
    data = FeatureDataset(dataset_dir)
    items = data.split("test")[:8]
    progress: list[int] = []
    metrics, records = evaluate(pipeline, data, items, progress=lambda done, total: progress.append(done))
    assert progress == list(range(1, 9)) and len(records) == 8
    for key in ("per", "class_er", "wer_first_pass", "wer_oracle", "wer", "cer", "real_time_factor"):
        assert metrics[key] >= 0
    assert metrics["wer_oracle"] <= metrics["wer"] + 1e-9  # the oracle is a lower bound
    assert metrics["class_er"] <= metrics["per"] + 0.2
    assert metrics["utterances"] == 8 and metrics["llm"] == "none" and "sim/s02" in metrics["wer_by_session"]
    text = format_metrics(metrics, ["a note"])
    assert "phoneme error rate" in text and "note: a note" in text
    assert len(sample_lines(records, 3)) == 3
    references, recognised = phoneme_sequences(pipeline, data, items)
    assert references[0] == items[0].phonemes and len(recognised) == 8


def test_confusion_driven_class_refinement(inventory):
    rng = np.random.default_rng(0)
    symbols = inventory.phonemes
    references, hypotheses = [], []
    for _ in range(400):  # a recogniser that cannot tell S from SH, nor any voicing pair
        reference = [str(s) for s in rng.choice(["S", "SH", "T", "D", "AA", "IY", "L", "M"], size=6)]
        swap = {"S": ["S", "SH"], "SH": ["S", "SH"], "T": ["T", "D"], "D": ["T", "D"]}
        hypotheses.append([str(rng.choice(swap[s])) if s in swap else s for s in reference])
        references.append(reference)
    conf = confusion(references, hypotheses, symbols)
    assert conf.support[symbols.index("AA")] > 100 and conf.recall()[symbols.index("AA")] == 1.0
    assert conf.recall()[symbols.index("S")] == pytest.approx(0.5, abs=0.1)
    assert {(a, b) for a, b, _, _ in conf.top_confusions(4)} == {("S", "SH"), ("SH", "S"), ("T", "D"), ("D", "T")}
    sim = similarity(conf)
    assert sim[symbols.index("S"), symbols.index("SH")] > 0.4 > sim[symbols.index("S"), symbols.index("T")]

    mapping = refine_classes(conf, inventory, threshold=0.25, min_support=20)
    assert mapping["S"] == mapping["SH"] and mapping["T"] == mapping["D"]
    assert mapping["S"] != mapping["T"] and mapping["AA"] != mapping["IY"]
    assert mapping["L"] != mapping["M"]  # separable phonemes get their own class
    assert mapping["P"] == "C_BILABIAL"  # unseen phonemes keep the prior
    refined = inventory.with_class_map(mapping)
    assert refined.phonemes == inventory.phonemes and "C_S_SH" in refined.classes
    assert "S SH" in describe_classes(mapping)
    deletions = confusion([["S", "T"]], [["S"]], symbols)
    assert (
        deletions.deletions[symbols.index("T")] == 1 and confusion([["S"]], [["S", "T"]], symbols).insertions.sum() == 1
    )


def test_fusion_weight_tuning():
    def record(reference: str, options: list[tuple[str, float, float, float]]) -> UtteranceRecord:
        candidates = [
            Candidate(tuple(text.split()), "beam", am=am, ngram=ngram, llm=llm) for text, am, ngram, llm in options
        ]
        return UtteranceRecord("u", "s", tuple(reference.split()), candidates[0].words, candidates[0].words, candidates)

    # The n-gram is misleading here and the LLM is right; tuning should find that out.
    records = [
        record("i met the cat", [("i bet the cat", -10, -3, -20), ("i met the cat", -10, -6, -5)]),
        record("pet the dog", [("bet the dog", -8, -2, -18), ("pet the dog", -8, -5, -4)]),
        record("good morning", [("good morning", -5, -2, -3), ("could morning", -5, -4, -15)]),
    ]
    start = FusionConfig(w_ngram=1.5, w_llm=0.0)
    tuned, before, after = tune_fusion(records, start, trials=300)
    assert before == pytest.approx(2 / 9) and after == 0.0
    assert tuned.w_llm > 0
    overrides = as_overrides(0.25, BeamConfig(), tuned)
    assert overrides["fusion"]["tau"] == 0.25 and overrides["fusion"]["w_llm"] == tuned.w_llm
    assert tune_fusion([], start) == (start, 0.0, 0.0)


def test_first_pass_tuning_runs(pipeline, dataset_dir):
    data = FeatureDataset(dataset_dir)
    lines: list[str] = []
    tau, beam, wer = tune_first_pass(pipeline, data, data.split("val"), max_utterances=4, report=lines.append)
    assert 0.0 <= tau <= 1.0 and beam.lm_weight > 0 and wer >= 0 and any("tau=" in line for line in lines)


# ------------------------------------------------------------------- commands
def test_cli_eval_decode_tune_inventory(bundle_dir, dataset_dir, sim_root, tmp_path, capsys):
    assert (
        main(
            [
                "eval",
                "--bundle",
                str(bundle_dir),
                "--data",
                str(dataset_dir),
                "--limit",
                "4",
                "--no-llm",
                "--out",
                str(tmp_path / "m.json"),
            ]
        )
        == 0
    )
    assert json.loads((tmp_path / "m.json").read_text())["utterances"] == 4
    assert "WER final" in capsys.readouterr().out

    assert (
        main(
            [
                "decode",
                "--bundle",
                str(bundle_dir),
                "--session",
                str(sim_root / "sim" / "s02"),
                "--limit",
                "3",
                "--no-llm",
                "--verbose-candidates",
                "1",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert out.count("phonemes:") == 3 and "calibration full" in out

    assert (
        main(
            [
                "tune",
                "--bundle",
                str(bundle_dir),
                "--data",
                str(dataset_dir),
                "--limit",
                "3",
                "--no-llm",
                "--skip-first-pass",
                "--out",
                str(tmp_path / "tuned.yaml"),
            ]
        )
        == 0
    )
    assert "fusion" in (tmp_path / "tuned.yaml").read_text()
    assert (
        main(
            [
                "eval",
                "--bundle",
                str(bundle_dir),
                "--data",
                str(dataset_dir),
                "--limit",
                "2",
                "--no-llm",
                "-c",
                str(tmp_path / "tuned.yaml"),
            ]
        )
        == 0
    )

    assert main(["inventory", "show"]) == 0
    assert "C_BILABIAL" in capsys.readouterr().out
    assert (
        main(["inventory", "confusions", "--bundle", str(bundle_dir), "--data", str(dataset_dir), "--limit", "6"]) == 0
    )
    assert (
        main(
            [
                "inventory",
                "refine",
                "--bundle",
                str(bundle_dir),
                "--data",
                str(dataset_dir),
                "--out",
                str(tmp_path / "classes.yaml"),
            ]
        )
        == 0
    )
    assert Inventory.from_yaml(tmp_path / "classes.yaml").phonemes == Bundle.load(bundle_dir).inventory.phonemes
