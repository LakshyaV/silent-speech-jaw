"""Confusion networks ("sausages") from N-best lists.

When several words are indistinguishable to the sensor, the number of plausible
sentences grows multiplicatively with sentence length and an N-best list can
only hold a sliver of them. A confusion network factorises that space: each slot
holds the alternative words competing for one position, with posteriors. It is
used in two ways:

* its k best paths under the n-gram LM recover good sentences that fell out of
  the beam;
* it is compiled into a grammar that constrains LLM generation, so the LLM
  searches the whole space while only ever choosing sensor-supported words.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ..eval.metrics import align
from .beam import Hypothesis
from .ngram import NGramLM

EPS = ""  # the "no word here" alternative


@dataclass
class Slot:
    """Alternatives for one word position: ``{word: posterior}`` (``EPS`` = skip)."""

    words: dict[str, float]

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.words.items(), key=lambda item: item[1], reverse=True)

    @property
    def optional(self) -> bool:
        return EPS in self.words

    def choices(self) -> list[str]:
        return [word for word, _ in self.ranked() if word != EPS]


def posteriors(hypotheses: Sequence[Hypothesis], scale: float = 1.0) -> list[float]:
    """Normalised posterior of each hypothesis from its fused score."""
    if not hypotheses:
        return []
    peak = max(h.score for h in hypotheses)
    weights = [math.exp(scale * (h.score - peak)) for h in hypotheses]
    total = sum(weights)
    return [w / total for w in weights]


def confusion_network(
    hypotheses: Sequence[Hypothesis],
    *,
    scale: float = 0.5,
    max_alternatives: int = 6,
    min_posterior: float = 0.01,
) -> list[Slot]:
    """Align every hypothesis to the best one and collect per-position alternatives.

    ``scale`` < 1 flattens the posteriors: the decoder's scores are over-confident
    because frame scores are not independent.
    """
    if not hypotheses:
        return []
    pivot = list(hypotheses[0].words)
    weights = posteriors(hypotheses, scale)
    substitutions: list[dict[str, float]] = [{} for _ in pivot]
    insertions: list[dict[str, float]] = [{} for _ in range(len(pivot) + 1)]
    for hypothesis, weight in zip(hypotheses, weights):
        position = 0
        inserted_here = False
        for op, _ref, hyp in align(pivot, list(hypothesis.words)):
            if op == "I":
                if not inserted_here:  # one inserted word per gap
                    bucket = insertions[position]
                    bucket[hyp] = bucket.get(hyp, 0.0) + weight
                    inserted_here = True
                continue
            bucket = substitutions[position]
            word = EPS if op == "D" else hyp
            bucket[word] = bucket.get(word, 0.0) + weight
            position += 1
            inserted_here = False

    def trim(bucket: dict[str, float]) -> dict[str, float]:
        ranked = sorted(bucket.items(), key=lambda item: item[1], reverse=True)
        kept = [(w, p) for i, (w, p) in enumerate(ranked[:max_alternatives]) if i == 0 or p >= min_posterior]
        return dict(kept)

    slots: list[Slot] = []
    for position in range(len(pivot) + 1):
        extra = insertions[position]
        mass = sum(extra.values())
        if mass >= min_posterior:
            bucket = dict(extra)
            bucket[EPS] = max(1.0 - mass, 1e-6)
            slots.append(Slot(trim(bucket)))
        if position < len(pivot):
            slots.append(Slot(trim(substitutions[position])))
    return slots


def best_paths(
    slots: Sequence[Slot],
    lm: NGramLM,
    *,
    k: int = 20,
    lm_weight: float = 1.0,
    slot_weight: float = 1.0,
    word_bonus: float = 0.0,
    beam: int = 64,
) -> list[tuple[tuple[str, ...], float]]:
    """The ``k`` best word sequences through a confusion network under the n-gram LM."""
    paths: list[tuple[float, tuple[str, ...], tuple[int, ...]]] = [(0.0, (), lm.bos_state())]
    for slot in slots:
        expanded: dict[tuple[str, ...], tuple[float, tuple[int, ...]]] = {}
        for score, words, state in paths:
            for word, posterior in slot.words.items():
                value = score + slot_weight * math.log(max(posterior, 1e-9))
                if word == EPS:
                    new_words, new_state = words, state
                else:
                    word_id = lm.word_id(word)
                    value += lm_weight * lm.logp(word_id, state) + word_bonus
                    new_words, new_state = (*words, word), lm.next_state(state, word_id)
                old = expanded.get(new_words)
                if old is None or value > old[0]:
                    expanded[new_words] = (value, new_state)
        ranked = sorted(expanded.items(), key=lambda item: item[1][0], reverse=True)[:beam]
        paths = [(value, words, state) for words, (value, state) in ranked]
    finals = []
    for score, words, state in paths:
        from .ngram import EOS_ID

        finals.append((words, score + lm_weight * lm.logp(EOS_ID, state)))
    finals.sort(key=lambda item: item[1], reverse=True)
    return finals[:k]


def describe(slots: Sequence[Slot], limit: int = 4) -> str:
    """Compact human-readable rendering, e.g. ``pat|bat|mat  the|a  (on)``."""
    parts = []
    for slot in slots:
        text = "|".join(slot.choices()[:limit])
        parts.append(f"({text})" if slot.optional else text)
    return "  ".join(parts)
