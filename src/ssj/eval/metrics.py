"""Edit-distance based error rates and alignments.

PER (phoneme), CER-class (articulatory class) and WER (word) are all the same
computation over different token sequences. The alignment is also what the
confusion analysis and the N-best confusion networks are built from.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

Token = Hashable
#: Alignment operations: ("=", ref, hyp), ("S", ref, hyp), ("D", ref, None), ("I", None, hyp)
Op = tuple[str, Token | None, Token | None]


def edit_distance(ref: Sequence[Token], hyp: Sequence[Token]) -> int:
    """Levenshtein distance with unit costs."""
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1]


def align(ref: Sequence[Token], hyp: Sequence[Token]) -> list[Op]:
    """Minimum-edit alignment of ``hyp`` against ``ref`` as a list of operations."""
    n, m = len(ref), len(hyp)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        row, above = cost[i], cost[i - 1]
        r = ref[i - 1]
        for j in range(1, m + 1):
            row[j] = min(above[j] + 1, row[j - 1] + 1, above[j - 1] + (r != hyp[j - 1]))
    ops: list[Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("=" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("I", None, hyp[j - 1]))
            j -= 1
    return ops[::-1]


@dataclass
class ErrorRate:
    """Accumulates edits over a test set."""

    errors: int = 0
    reference_tokens: int = 0
    sentences: int = 0
    sentence_errors: int = 0

    def add(self, ref: Sequence[Token], hyp: Sequence[Token]) -> int:
        distance = edit_distance(ref, hyp)
        self.errors += distance
        self.reference_tokens += len(ref)
        self.sentences += 1
        self.sentence_errors += int(distance > 0)
        return distance

    @property
    def rate(self) -> float:
        return self.errors / max(self.reference_tokens, 1)

    @property
    def sentence_rate(self) -> float:
        return self.sentence_errors / max(self.sentences, 1)

    def __float__(self) -> float:
        return self.rate


def error_rate(refs: Sequence[Sequence[Token]], hyps: Sequence[Sequence[Token]]) -> float:
    total = ErrorRate()
    for ref, hyp in zip(refs, hyps):
        total.add(ref, hyp)
    return total.rate


def character_error_rate(refs: Sequence[Sequence[str]], hyps: Sequence[Sequence[str]]) -> float:
    """CER over the space-joined word sequences."""
    return error_rate([list(" ".join(r)) for r in refs], [list(" ".join(h)) for h in hyps])


def oracle_error_rate(refs: Sequence[Sequence[Token]], nbests: Sequence[Sequence[Sequence[Token]]]) -> float:
    """Error rate if the best hypothesis in each N-best list were always chosen.

    The gap between this and the 1-best error rate is the headroom available to
    rescoring: no language model can do better than the oracle.
    """
    total = ErrorRate()
    for ref, nbest in zip(refs, nbests):
        best = min(nbest, key=lambda hyp: edit_distance(ref, hyp)) if nbest else []
        total.add(ref, best)
    return total.rate
