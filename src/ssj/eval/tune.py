"""Tuning of decoder weights on held-out data.

Two groups of parameters are tuned, in this order:

first pass  ``tau`` (how much within-class phoneme detail to trust), the n-gram
            weight and the word bonus. Each trial re-runs the beam search, so a
            coordinate search over a subset of utterances is used.
fusion      the weights that combine CTC, n-gram and LLM scores. Candidate
            scores are computed once and re-ranked for every trial, so a dense
            search is cheap.

``tau`` is the interesting one: its tuned value is a direct measurement of how
much phonetic detail beyond the articulatory class the sensor delivers.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import partial
from typing import Any

import numpy as np

from ..data.build import FeatureDataset, Item
from ..decode.beam import BeamConfig, beam_search
from ..decode.ctc_score import tie_classes
from ..llm.fusion import FusionConfig
from ..runtime.pipeline import Pipeline
from .metrics import ErrorRate, edit_distance
from .report import UtteranceRecord

TAU_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
LM_WEIGHT_GRID = (0.4, 0.7, 0.9, 1.2, 1.6)
WORD_BONUS_GRID = (0.0, 0.5, 1.0, 2.0, 3.0)


def first_pass_wer(
    pipeline: Pipeline,
    log_probs: Sequence[np.ndarray],
    references: Sequence[Sequence[str]],
    tau: float,
    beam: BeamConfig,
) -> float:
    decoder = pipeline.decoder
    total = ErrorRate()
    for lp, reference in zip(log_probs, references):
        nbest = beam_search(tie_classes(lp, decoder.inventory, tau), decoder.trie, decoder.lm, beam)
        total.add(list(reference), list(nbest[0].words) if nbest else [])
    return total.rate


def tune_first_pass(
    pipeline: Pipeline,
    data: FeatureDataset,
    items: Sequence[Item],
    *,
    max_utterances: int = 80,
    report: Callable[[str], None] | None = None,
) -> tuple[float, BeamConfig, float]:
    """Coordinate search over ``tau``, ``lm_weight`` and ``word_bonus``; returns the best and its WER."""
    items = list(items)[:max_utterances]
    log_probs = [pipeline.bundle.log_probs(data.load(item)) for item in items]
    references = [item.text.split() for item in items]
    tau = pipeline.decoder.fusion.tau
    beam = pipeline.decoder.beam
    best = first_pass_wer(pipeline, log_probs, references, tau, beam)

    def search(name: str, grid: Sequence[float], current: float, run: Callable[[float], float]) -> float:
        nonlocal best
        chosen = current
        for value in grid:
            if value == current:
                continue
            score = run(value)
            if report:
                report(f"  {name}={value:<5} WER {score:.3f}")
            if score < best - 1e-9:
                best, chosen = score, value
        return chosen

    def run_with(tau_value: float, beam_value: BeamConfig) -> float:
        return first_pass_wer(pipeline, log_probs, references, tau_value, beam_value)

    for _ in range(2):  # two sweeps: the parameters interact mildly
        tau = search("tau", TAU_GRID, tau, partial(lambda b, v: run_with(v, b), beam))
        weight = search(
            "lm_weight",
            LM_WEIGHT_GRID,
            beam.lm_weight,
            partial(lambda t, b, v: run_with(t, replace(b, lm_weight=v)), tau, beam),
        )
        beam = replace(beam, lm_weight=weight)
        bonus = search(
            "word_bonus",
            WORD_BONUS_GRID,
            beam.word_bonus,
            partial(lambda t, b, v: run_with(t, replace(b, word_bonus=v)), tau, beam),
        )
        beam = replace(beam, word_bonus=bonus)
    return tau, beam, best


def tune_fusion(
    records: Sequence[UtteranceRecord], base: FusionConfig, *, trials: int = 600, seed: int = 0
) -> tuple[FusionConfig, float, float]:
    """Search fusion weights on cached candidates; returns (config, WER before, WER after)."""
    features, errors, lengths = [], [], 0
    for record in records:
        if not record.candidates:
            lengths += len(record.reference)
            continue
        features.append(np.array([c.features() for c in record.candidates]))
        errors.append(np.array([edit_distance(record.reference, c.words) for c in record.candidates]))
        lengths += len(record.reference)
    empty_errors = sum(len(r.reference) for r in records if not r.candidates)

    def wer(weights: np.ndarray) -> float:
        total = empty_errors
        for f, e in zip(features, errors):
            total += int(e[int(np.argmax(f @ weights))])
        return total / max(lengths, 1)

    start = np.array([base.w_am, base.w_ngram, base.w_llm, base.w_words])
    before = wer(start)
    best_weights, best = start, before
    if not features:
        return base, before, before
    rng = np.random.default_rng(seed)
    uses_llm = any(np.any(f[:, 2] != 0.0) for f in features)
    for _ in range(trials):
        trial = np.array(
            [1.0, rng.uniform(0.0, 1.5), rng.uniform(0.0, 1.5) if uses_llm else 0.0, rng.uniform(-1.0, 4.0)]
        )
        value = wer(trial)
        if value < best - 1e-9:
            best_weights, best = trial, value
    tuned = replace(
        base,
        w_am=float(best_weights[0]),
        w_ngram=float(round(best_weights[1], 3)),
        w_llm=float(round(best_weights[2], 3)) if uses_llm else base.w_llm,
        w_words=float(round(best_weights[3], 3)),
    )
    return tuned, before, best


def as_overrides(tau: float, beam: BeamConfig, fusion: FusionConfig) -> dict[str, Any]:
    """Tuned values as a config fragment for ``ssj decode`` / ``ssj live``."""
    return {
        "beam": {"lm_weight": beam.lm_weight, "word_bonus": beam.word_bonus},
        "fusion": {
            "tau": tau,
            "w_am": fusion.w_am,
            "w_ngram": fusion.w_ngram,
            "w_llm": fusion.w_llm,
            "w_words": fusion.w_words,
        },
    }
