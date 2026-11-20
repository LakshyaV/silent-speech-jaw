"""Lexicon-constrained CTC beam search with n-gram fusion.

This turns frame-level phoneme scores into ranked word sequences. A hypothesis
is a path through the pronunciation trie plus the words completed so far; at
every frame it may emit blank, repeat its last phoneme, or advance to a child
node, and whenever it reaches the end of a word it may commit that word, pay its
language-model cost and return to the trie root.

Hypotheses are identified by (word sequence, trie node, last emitted symbol),
and hypotheses reaching the same identity along different alignments are merged
by log-add, so a hypothesis' acoustic score is the total probability of its
alignments rather than that of the single best one.

The search is pure Python over a few hundred hypotheses per frame. Two things
keep it fast: frames where blank is near-certain are consumed in one step, and
only the handful of phonemes with non-negligible score are expanded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..phonology import BLANK_ID
from .ngram import EOS_ID, NGramLM
from .trie import ROOT, LexiconTrie


@dataclass(frozen=True)
class BeamConfig:
    beam_size: int = 300
    beam_threshold: float = 22.0  # hypotheses this far below the best are dropped
    token_beam: float = 9.0  # phonemes this far below the best phoneme of a frame are ignored
    max_tokens: int = 8  # at most this many phonemes expanded per frame
    blank_skip: float = 0.9995  # blank posterior above which a frame is consumed as blank
    lm_weight: float = 0.9
    word_bonus: float = 1.0  # added per word; offsets the LM's preference for fewer words
    nbest: int = 40


@dataclass
class Hypothesis:
    words: tuple[str, ...]
    word_ids: tuple[int, ...]
    am: float  # acoustic log-score, summed over alignments
    lm: float  # n-gram log-probability including </s>
    score: float  # am + lm_weight * lm + word_bonus * len(words)


def _logadd(a: float, b: float) -> float:
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def beam_search(
    scores: np.ndarray, trie: LexiconTrie, lm: NGramLM, config: BeamConfig | None = None
) -> list[Hypothesis]:
    """Decode frame scores ``[T, V]`` (log domain, blank at index 0) into an N-best list."""
    cfg = config or BeamConfig()
    alpha, beta = cfg.lm_weight, cfg.word_bonus
    children, words_at, lookahead = trie.children, trie.words, trie.lookahead
    skip_log = math.log(cfg.blank_skip)

    # Word-sequence store: a hypothesis refers to its words by one integer.
    seq_parent, seq_word = [-1], [-1]
    seq_lm = [0.0]
    seq_state = [lm.bos_state()]
    seq_len = [0]
    seq_index: dict[tuple[int, int], int] = {}

    def extend(seq: int, word: int) -> int:
        key = (seq, word)
        found = seq_index.get(key)
        if found is None:
            found = len(seq_parent)
            seq_index[key] = found
            seq_parent.append(seq)
            seq_word.append(word)
            seq_lm.append(seq_lm[seq] + lm.logp(word, seq_state[seq]))
            seq_state.append(lm.next_state(seq_state[seq], word))
            seq_len.append(seq_len[seq] + 1)
        return found

    hyps: dict[tuple[int, int, int], float] = {(0, ROOT, BLANK_ID): 0.0}
    order = np.argsort(-scores[:, 1:], axis=1)[:, : cfg.max_tokens] + 1
    for t in range(len(scores)):
        row = scores[t].tolist()
        blank = row[BLANK_ID]
        new: dict[tuple[int, int, int], float] = {}
        if blank >= skip_log:
            for (seq, node, _last), value in hyps.items():
                key = (seq, node, BLANK_ID)
                value += blank
                old = new.get(key)
                new[key] = value if old is None else _logadd(old, value)
            hyps = new
            continue

        candidates = order[t].tolist()
        floor = row[candidates[0]] - cfg.token_beam
        candidates = [k for k in candidates if row[k] >= floor]
        for (seq, node, last), value in hyps.items():
            key = (seq, node, BLANK_ID)
            v = value + blank
            old = new.get(key)
            new[key] = v if old is None else _logadd(old, v)
            if last != BLANK_ID:
                key = (seq, node, last)
                v = value + row[last]
                old = new.get(key)
                new[key] = v if old is None else _logadd(old, v)
            below = children[node]
            if not below:
                continue
            for k in candidates:
                if k == last:
                    continue  # a repeated phoneme needs a blank in between
                child = below.get(k)
                if child is None:
                    continue
                v = value + row[k]
                key = (seq, child, k)
                old = new.get(key)
                new[key] = v if old is None else _logadd(old, v)
                for word in words_at[child]:
                    key = (extend(seq, word), ROOT, k)
                    old = new.get(key)
                    new[key] = v if old is None else _logadd(old, v)

        # Prune on the fused score, with LM lookahead for words still in progress.
        ranked = [
            (value + alpha * (seq_lm[key[0]] + lookahead[key[1]]) + beta * seq_len[key[0]], key, value)
            for key, value in new.items()
        ]
        best = max(item[0] for item in ranked)
        cutoff = best - cfg.beam_threshold
        ranked = [item for item in ranked if item[0] >= cutoff]
        if len(ranked) > cfg.beam_size:
            ranked.sort(key=lambda item: item[0], reverse=True)
            ranked = ranked[: cfg.beam_size]
        hyps = {key: value for _, key, value in ranked}

    # Only hypotheses at a word boundary are complete.
    finished: dict[int, float] = {}
    for (seq, node, _last), value in hyps.items():
        if node == ROOT:
            old = finished.get(seq)
            finished[seq] = value if old is None else _logadd(old, value)
    results: list[Hypothesis] = []
    for seq, am in finished.items():
        ids: list[int] = []
        cursor = seq
        while cursor > 0:
            ids.append(seq_word[cursor])
            cursor = seq_parent[cursor]
        ids.reverse()
        total_lm = seq_lm[seq] + lm.logp(EOS_ID, seq_state[seq])
        results.append(
            Hypothesis(
                tuple(lm.vocab[i] for i in ids), tuple(ids), am, total_lm, am + alpha * total_lm + beta * len(ids)
            )
        )
    results.sort(key=lambda h: h.score, reverse=True)
    return results[: cfg.nbest]
