"""CTC posterior utilities: greedy path, class tying, exact sequence scoring.

Everything here is plain numpy so it runs wherever the exported model runs.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..phonology import BLANK_ID, Inventory

NEG_INF = -1e30


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def greedy_decode(log_probs: np.ndarray) -> list[int]:
    """Best-path CTC decoding of ``[T, V]`` scores: argmax, collapse repeats, drop blanks."""
    path = np.asarray(log_probs).argmax(axis=-1)
    if len(path) == 0:
        return []
    keep = np.concatenate(([True], path[1:] != path[:-1])) & (path != BLANK_ID)
    return [int(t) for t in path[keep]]


def marginalize(log_probs: np.ndarray, inventory: Inventory) -> np.ndarray:
    """Phoneme log-posteriors ``[T, V]`` -> articulatory-class log-posteriors ``[T, K]``."""
    peak = log_probs.max(axis=-1, keepdims=True)
    probs = np.exp(log_probs - peak)
    return np.log(probs @ inventory.membership.astype(probs.dtype) + 1e-30) + peak


def tie_classes(log_probs: np.ndarray, inventory: Inventory, tau: float) -> np.ndarray:
    """Blend phoneme scores with their class scores.

    ``tau = 1`` returns the phoneme posteriors unchanged. ``tau = 0`` gives every
    phoneme the score of its whole class, i.e. the model's evidence that *some*
    member of the class is being articulated: words that differ only inside a
    class (pat / bat / mat) then tie acoustically and are decided by the
    language models. In between, the within-class preference is trusted
    partially. ``tau`` is tuned on held-out data, which is how the system finds
    out how much phonetic detail the sensor really delivers.
    """
    if tau >= 1.0:
        return log_probs
    class_scores = marginalize(log_probs, inventory)[:, inventory.class_ids]
    tied = (1.0 - tau) * class_scores + tau * log_probs
    tied[:, BLANK_ID] = log_probs[:, BLANK_ID]
    return tied


def ctc_forward(scores: np.ndarray, tokens: Sequence[int]) -> float:
    """Log-likelihood of a token sequence under frame scores ``[T, V]`` (all alignments).

    This is the exact CTC forward algorithm. It lets any candidate sentence -
    from the beam search, from the confusion network or proposed freely by the
    LLM - be checked against the sensor evidence on equal terms.
    """
    T = len(scores)
    L = len(tokens)
    if T == 0:
        return 0.0 if L == 0 else NEG_INF
    if L == 0:
        return float(scores[:, BLANK_ID].sum())
    extended = np.full(2 * L + 1, BLANK_ID, dtype=np.int64)
    extended[1::2] = np.asarray(tokens, dtype=np.int64)
    S = len(extended)
    if T < L:
        return NEG_INF
    # A skip over a blank is allowed only between two different labels.
    can_skip = np.zeros(S, dtype=bool)
    can_skip[3::2] = extended[3::2] != extended[1:-2:2]

    emissions = scores[:, extended]  # [T, S]
    alpha = np.full(S, NEG_INF)
    alpha[0] = emissions[0, 0]
    alpha[1] = emissions[0, 1]
    for t in range(1, T):
        stay = alpha
        step = np.concatenate(([NEG_INF], alpha[:-1]))
        skip = np.where(can_skip, np.concatenate(([NEG_INF, NEG_INF], alpha[:-2])), NEG_INF)
        peak = np.maximum(np.maximum(stay, step), skip)
        total = np.exp(stay - peak) + np.exp(step - peak) + np.exp(skip - peak)
        alpha = peak + np.log(total) + emissions[t]
        alpha = np.maximum(alpha, NEG_INF)
    return float(np.logaddexp(alpha[-1], alpha[-2]))


def sequence_score(scores: np.ndarray, pronunciations: Sequence[Sequence[Sequence[int]]]) -> tuple[float, list[int]]:
    """Best CTC score of a word sequence over each word's pronunciation variants.

    ``pronunciations[i]`` lists the variants (token id sequences) of word *i*.
    Variants are chosen by coordinate ascent starting from each word's first
    pronunciation, which is exact for the common case of one ambiguous word and
    never worse than using first pronunciations throughout.
    """
    choice = [0] * len(pronunciations)

    def flat() -> list[int]:
        return [t for word, variant in zip(pronunciations, choice) for t in word[variant]]

    best = ctc_forward(scores, flat())
    for index, variants in enumerate(pronunciations):
        if len(variants) < 2:
            continue
        for variant in range(1, len(variants)):
            previous = choice[index]
            choice[index] = variant
            value = ctc_forward(scores, flat())
            if value > best:
                best = value
            else:
                choice[index] = previous
    return best, choice
