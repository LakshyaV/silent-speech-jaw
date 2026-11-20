"""Phoneme confusion analysis and data-driven refinement of the class inventory.

The built-in articulatory classes are a phonetic prior about what a jaw sensor
can separate. Once a recogniser exists, its confusions measure that directly:
phonemes that are confused with each other about as often as they are
recognised belong in one class, whatever the textbook says, and phonemes the
model separates reliably deserve their own. :func:`refine_classes` re-derives
the phoneme -> class map by agglomerative clustering of the confusion matrix;
retraining with the refined map sharpens both the auxiliary loss and decoding.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..phonology import Inventory
from .metrics import align


@dataclass
class Confusion:
    symbols: tuple[str, ...]
    matrix: np.ndarray  # [V, V] counts, row = reference, column = recognised
    deletions: np.ndarray  # [V]
    insertions: np.ndarray  # [V]

    @property
    def support(self) -> np.ndarray:
        """Number of reference occurrences of each symbol."""
        return self.matrix.sum(axis=1) + self.deletions

    def recall(self) -> np.ndarray:
        return np.diag(self.matrix) / np.maximum(self.support, 1)

    def top_confusions(self, k: int = 15) -> list[tuple[str, str, int, float]]:
        """Most frequent substitutions as ``(reference, recognised, count, rate)``."""
        off = self.matrix.copy()
        np.fill_diagonal(off, 0)
        support = np.maximum(self.support, 1)
        pairs = [
            (self.symbols[i], self.symbols[j], int(off[i, j]), float(off[i, j] / support[i]))
            for i, j in zip(*np.nonzero(off))
        ]
        return sorted(pairs, key=lambda p: p[2], reverse=True)[:k]


def confusion(
    references: Sequence[Sequence[str]], hypotheses: Sequence[Sequence[str]], symbols: Sequence[str]
) -> Confusion:
    """Accumulate substitution, deletion and insertion counts from aligned sequences."""
    index = {s: i for i, s in enumerate(symbols)}
    n = len(symbols)
    matrix = np.zeros((n, n), dtype=np.int64)
    deletions = np.zeros(n, dtype=np.int64)
    insertions = np.zeros(n, dtype=np.int64)
    for reference, hypothesis in zip(references, hypotheses):
        for op, ref, hyp in align(list(reference), list(hypothesis)):
            if op == "D":
                deletions[index[ref]] += 1
            elif op == "I":
                insertions[index[hyp]] += 1
            else:
                matrix[index[ref], index[hyp]] += 1
    return Confusion(tuple(symbols), matrix, deletions, insertions)


def similarity(conf: Confusion, smoothing: float = 1.0) -> np.ndarray:
    """Symmetric confusability in [0, 1]: 1 means interchangeable, 0 never confused.

    For a pair (i, j) it compares how often the two are swapped with how often
    each is recognised as itself, so two phonemes the model cannot separate
    score near 1 regardless of how frequent they are.
    """
    m = conf.matrix.astype(np.float64)
    swapped = m + m.T
    own = np.diag(m)[:, None] + np.diag(m)[None, :]
    sim = swapped / (swapped + own + smoothing)
    np.fill_diagonal(sim, 1.0)
    return sim


def refine_classes(
    conf: Confusion,
    inventory: Inventory,
    *,
    threshold: float = 0.25,
    min_support: int = 20,
) -> dict[str, str]:
    """Cluster phonemes by confusability; returns a phoneme -> class-name map.

    Average-linkage agglomerative clustering, merging while the mean pairwise
    similarity between two clusters is at least ``threshold``. Vowels,
    consonants and gestures are never merged with each other. Phonemes seen
    fewer than ``min_support`` times keep their current class: there is not
    enough evidence to move them.
    """
    symbols = list(conf.symbols)
    sim = similarity(conf)
    support = conf.support
    kind = dict(zip(inventory.phonemes, inventory.kinds))
    current = {p: inventory.classes[c - 1] for p, c in zip(inventory.phonemes, inventory.class_of)}

    movable = [i for i, s in enumerate(symbols) if support[i] >= min_support]
    clusters: list[list[int]] = [[i] for i in movable]
    while len(clusters) > 1:
        best, pair = threshold, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                if kind[symbols[clusters[a][0]]] != kind[symbols[clusters[b][0]]]:
                    continue
                value = float(np.mean([sim[i, j] for i in clusters[a] for j in clusters[b]]))
                if value >= best:
                    best, pair = value, (a, b)
        if pair is None:
            break
        a, b = pair
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]

    mapping: dict[str, str] = {}
    for cluster in clusters:
        members = sorted(symbols[i] for i in cluster)
        prefix = {"vowel": "V", "consonant": "C", "gesture": "G"}[kind[members[0]]]
        name = f"{prefix}_{'_'.join(members)}"
        for member in members:
            mapping[member] = name
    for phoneme in inventory.phonemes:  # rare or unseen phonemes keep their class
        mapping.setdefault(phoneme, current[phoneme])
    return mapping


def describe_classes(mapping: dict[str, str]) -> str:
    groups: dict[str, list[str]] = {}
    for phoneme, name in mapping.items():
        groups.setdefault(name, []).append(phoneme)
    return "\n".join(f"  {' '.join(sorted(members))}" for _, members in sorted(groups.items()))
