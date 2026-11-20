"""End-to-end evaluation of a deployable bundle on a built dataset.

Evaluation always goes through the exported bundle and the runtime pipeline,
so the reported numbers describe exactly what will run on the device. Each
stage is scored separately, because they fail differently:

    PER / class ER   the recogniser alone (greedy)
    first-pass WER   beam search with the n-gram model
    oracle WER       best sentence present in the N-best: the ceiling for rescoring
    WER              the final fused output (with the LLM if one is configured)
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..data.build import FeatureDataset, Item
from ..decode.ctc_score import greedy_decode, marginalize
from ..llm.fusion import Candidate, DecodeResult
from ..runtime.pipeline import Pipeline
from .metrics import ErrorRate, edit_distance


@dataclass
class UtteranceRecord:
    """Everything needed to re-rank one utterance offline (used by the tuner)."""

    utt_id: str
    session: str
    reference: tuple[str, ...]
    hypothesis: tuple[str, ...]
    first_pass: tuple[str, ...]
    candidates: list[Candidate] = field(default_factory=list)
    n_frames: int = 0


def evaluate(
    pipeline: Pipeline,
    data: FeatureDataset,
    items: Sequence[Item],
    *,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, Any], list[UtteranceRecord]]:
    """Decode ``items`` and return (metrics, per-utterance records)."""
    inventory = pipeline.bundle.inventory
    if data.meta["frontend_fingerprint"] != pipeline.bundle.meta["frontend_fingerprint"]:
        raise ValueError("the dataset and the bundle were produced by different front-ends")
    per, class_er = ErrorRate(), ErrorRate()
    wer, first, oracle, cer = ErrorRate(), ErrorRate(), ErrorRate(), ErrorRate()
    by_session: dict[str, ErrorRate] = defaultdict(ErrorRate)
    by_source: dict[str, int] = defaultdict(int)
    records: list[UtteranceRecord] = []
    stage_time: dict[str, float] = defaultdict(float)
    audio_s = 0.0
    started = time.perf_counter()

    for number, item in enumerate(items):
        feats = data.load(item)
        log_probs = pipeline.bundle.log_probs(feats)
        reference_ids = inventory.encode(item.phonemes)
        per.add(reference_ids, greedy_decode(log_probs))
        class_er.add(inventory.to_class_ids(reference_ids), greedy_decode(marginalize(log_probs, inventory)))

        result: DecodeResult = pipeline.decoder.decode(log_probs)
        reference = tuple(item.text.split())
        wer.add(reference, result.words)
        first.add(reference, result.first_pass)
        cer.add(list(" ".join(reference)), list(" ".join(result.words)))
        by_session[item.session_key].add(reference, result.words)
        best = min(result.candidates, key=lambda c: edit_distance(reference, c.words), default=None)
        oracle.add(reference, best.words if best else ())
        if result.best is not None:
            by_source[result.best.source] += 1
        for stage, seconds in result.timings.items():
            stage_time[stage] += seconds
        audio_s += len(feats) / pipeline.bundle.frontend_config.fs
        records.append(
            UtteranceRecord(
                item.utt_id,
                item.session_key,
                reference,
                result.words,
                result.first_pass,
                result.candidates,
                len(log_probs),
            )
        )
        if progress:
            progress(number + 1, len(items))

    n = max(len(items), 1)
    elapsed = time.perf_counter() - started
    metrics = {
        "utterances": len(items),
        "per": per.rate,
        "class_er": class_er.rate,
        "wer_first_pass": first.rate,
        "wer_oracle": oracle.rate,
        "wer": wer.rate,
        "cer": cer.rate,
        "sentence_error_rate": wer.sentence_rate,
        "wer_by_session": {key: value.rate for key, value in sorted(by_session.items())},
        "winning_source": dict(by_source),
        "llm": pipeline.backend.name,
        "seconds_per_utterance": elapsed / n,
        "real_time_factor": elapsed / max(audio_s, 1e-9),
        "stage_seconds": {stage: seconds / n for stage, seconds in stage_time.items()},
    }
    return metrics, records


def format_metrics(metrics: dict[str, Any], notes: Sequence[str] = ()) -> str:
    lines = [
        f"utterances          {metrics['utterances']}",
        f"phoneme error rate  {metrics['per']:.3f}",
        f"class error rate    {metrics['class_er']:.3f}",
        f"WER first pass      {metrics['wer_first_pass']:.3f}   (beam search + n-gram)",
        f"WER oracle          {metrics['wer_oracle']:.3f}   (best candidate available)",
        f"WER final           {metrics['wer']:.3f}   (fusion, LLM: {metrics['llm']})",
        f"CER final           {metrics['cer']:.3f}",
        f"time per utterance  {1000 * metrics['seconds_per_utterance']:.0f} ms   (real-time factor {metrics['real_time_factor']:.2f})",
    ]
    if len(metrics["wer_by_session"]) > 1:
        lines.append("WER by session      " + ", ".join(f"{k} {v:.3f}" for k, v in metrics["wer_by_session"].items()))
    lines.extend(f"note: {note}" for note in notes)
    return "\n".join(lines)


def phoneme_sequences(
    pipeline: Pipeline, data: FeatureDataset, items: Sequence[Item]
) -> tuple[list[list[str]], list[list[str]]]:
    """Reference and greedy-recognised phoneme strings (input to confusion analysis)."""
    inventory = pipeline.bundle.inventory
    references = [list(item.phonemes) for item in items]
    recognised = [inventory.decode(greedy_decode(pipeline.bundle.log_probs(data.load(item)))) for item in items]
    return references, recognised


def sample_lines(records: Sequence[UtteranceRecord], k: int = 8, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(records), size=min(k, len(records)), replace=False) if records else []
    lines = []
    for index in sorted(int(i) for i in chosen):
        record = records[index]
        mark = "ok " if record.reference == record.hypothesis else "ERR"
        lines.append(f"[{mark}] ref: {' '.join(record.reference)}\n      hyp: {' '.join(record.hypothesis)}")
    return lines
