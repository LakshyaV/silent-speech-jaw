"""Pronunciation prefix tree used by the lexicon-constrained decoder."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from ..phonology import Inventory, Lexicon
from .ngram import BOS, EOS, UNK, NGramLM

ROOT = 0


class LexiconTrie:
    """Prefix tree over phoneme ids; nodes list the words whose pronunciation ends there.

    ``lookahead[node]`` is the best unigram log-probability of any word reachable
    from the node. Adding it to a partial word's priority lets the beam prune
    paths that can only lead to rare words long before the word is complete.
    """

    def __init__(self, entries: Iterable[tuple[int, Sequence[int]]], unigram_logp: np.ndarray | None = None):
        self.children: list[dict[int, int]] = [{}]
        self.words: list[list[int]] = [[]]
        parent: list[int] = [-1]
        n_entries = 0
        for word_id, tokens in entries:
            node = ROOT
            for token in tokens:
                nxt = self.children[node].get(token)
                if nxt is None:
                    nxt = len(self.children)
                    self.children[node][token] = nxt
                    self.children.append({})
                    self.words.append([])
                    parent.append(node)
                node = nxt
            if node != ROOT and word_id not in self.words[node]:
                self.words[node].append(word_id)
                n_entries += 1
        self.n_entries = n_entries
        lookahead = [-np.inf] * len(self.children)
        if unigram_logp is not None:
            for node in range(len(self.children) - 1, 0, -1):  # children always follow parents
                best = max((float(unigram_logp[w]) for w in self.words[node]), default=-np.inf)
                lookahead[node] = max(lookahead[node], best)
                lookahead[parent[node]] = max(lookahead[parent[node]], lookahead[node])
        else:
            lookahead = [0.0] * len(self.children)
        lookahead[ROOT] = 0.0
        self.lookahead = lookahead

    def __len__(self) -> int:
        return len(self.children)

    @classmethod
    def build(cls, lexicon: Lexicon, inventory: Inventory, lm: NGramLM) -> LexiconTrie:
        """Trie over every word that is both pronounceable and known to the language model."""
        unigram = lm.unigram_logp()

        def entries():
            for word, prons in lexicon.items():
                if word in (BOS, EOS, UNK) or word not in lm:
                    continue
                word_id = lm.word_id(word)
                for pron in prons:
                    if all(p in inventory for p in pron):
                        yield word_id, inventory.encode(pron)

        return cls(entries(), unigram)
