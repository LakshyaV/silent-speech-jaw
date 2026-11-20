"""Word n-gram language model: training, storage and lookup in pure numpy.

The n-gram is the first-pass language model of the decoder: cheap enough to be
consulted for every word hypothesis in the beam, and good enough to produce an
N-best list worth handing to the LLM. It is self-contained so that building a
model needs nothing but a text file::

    lm = train_ngram(sentences, order=3, max_vocab=30000)
    lm.save("assets/lm/en.npz")

Estimation is interpolated Kneser-Ney with one discount per order. N-grams are
stored as sorted packed integer keys with parallel probability and back-off
arrays (16 bytes per n-gram), so a multi-million n-gram model loads instantly
and fits comfortably beside the LLM on a small board. ARPA files can be
imported and exported for interoperability with KenLM and SRILM.
"""

from __future__ import annotations

import gzip
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from ..phonology.text import CONTROL_WORDS

BOS, EOS, UNK = "<s>", "</s>", "<unk>"
BOS_ID, EOS_ID, UNK_ID = 0, 1, 2
_LN10 = math.log(10.0)
_MISSING = -99.0


class NGramLM:
    """Back-off n-gram model over word ids. All scores are natural logarithms."""

    def __init__(self, vocab: Sequence[str], logp: list[np.ndarray], backoff: list[np.ndarray], keys: list[np.ndarray]):
        """``keys[k]`` are the sorted packed keys of the (k+1)-grams; ``keys[0]`` is unused."""
        self.vocab = list(vocab)
        self.order = len(logp)
        self._index = {word: i for i, word in enumerate(self.vocab)}
        self._size = len(self.vocab)
        # float32 tables: 16 bytes per n-gram including its key
        self._logp = [np.asarray(table, dtype=np.float32) for table in logp]
        self._backoff = [np.asarray(table, dtype=np.float32) for table in backoff]
        self._keys = keys
        self._unigram_logp = self._logp[0].tolist()
        self._unigram_backoff = self._backoff[0].tolist()
        self._cache: dict[tuple[tuple[int, ...], int], float] = {}
        # Control gestures are not language: they cost nothing and leave the context untouched,
        # so "turn it off <clench>" is scored exactly like "turn it off".
        self._transparent = frozenset(self._index[w] for w in CONTROL_WORDS if w in self._index)

    # ---------------------------------------------------------------- lookup
    def __contains__(self, word: object) -> bool:
        return word in self._index

    def word_id(self, word: str) -> int:
        return self._index.get(word, UNK_ID)

    def n_grams(self) -> list[int]:
        return [len(table) for table in self._logp]

    def bos_state(self) -> tuple[int, ...]:
        return (BOS_ID,)

    def next_state(self, state: tuple[int, ...], word_id: int) -> tuple[int, ...]:
        if word_id in self._transparent:
            return state
        return (*state, word_id)[-(self.order - 1) :] if self.order > 1 else ()

    def _find(self, ids: tuple[int, ...]) -> int:
        """Row of an n-gram (n >= 2) in its table, or -1."""
        key = 0
        for i in ids:
            key = key * self._size + i
        table = self._keys[len(ids) - 1]
        row = int(np.searchsorted(table, np.uint64(key)))
        return row if row < len(table) and int(table[row]) == key else -1

    def logp(self, word_id: int, state: tuple[int, ...]) -> float:
        """``log P(word | state)`` with back-off."""
        if word_id in self._transparent:
            return 0.0
        cached = self._cache.get((state, word_id))
        if cached is not None:
            return cached
        history = state[-(self.order - 1) :] if self.order > 1 else ()
        penalty = 0.0
        value = None
        while history:
            row = self._find((*history, word_id))
            if row >= 0:
                value = penalty + float(self._logp[len(history)][row])
                break
            if len(history) == 1:
                penalty += self._unigram_backoff[history[0]]
            else:
                context = self._find(history)
                if context >= 0:
                    penalty += float(self._backoff[len(history) - 1][context])
            history = history[1:]
        if value is None:
            value = penalty + self._unigram_logp[word_id]
        if len(self._cache) > 2_000_000:
            self._cache.clear()
        self._cache[(state, word_id)] = value
        return value

    def unigram_logp(self) -> np.ndarray:
        return self._logp[0]

    def score(self, words: Sequence[str], *, eos: bool = True) -> float:
        """Log-probability of a sentence (with ``<s>`` context and optional ``</s>``)."""
        state = self.bos_state()
        total = 0.0
        for word in words:
            word_id = self.word_id(word)
            total += self.logp(word_id, state)
            state = self.next_state(state, word_id)
        if eos:
            total += self.logp(EOS_ID, state)
        return total

    def perplexity(self, sentences: Iterable[Sequence[str]]) -> float:
        total, count = 0.0, 0
        for words in sentences:
            total += self.score(words)
            count += len(words) + 1
        return math.exp(-total / max(count, 1))

    # --------------------------------------------------------------- storage
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {"vocab": np.array(self.vocab, dtype=object), "order": np.array(self.order)}
        for k in range(self.order):
            arrays[f"logp{k}"] = self._logp[k]
            arrays[f"backoff{k}"] = self._backoff[k]
            arrays[f"keys{k}"] = self._keys[k]
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> NGramLM:
        with np.load(Path(path), allow_pickle=True) as data:
            order = int(data["order"])
            return cls(
                [str(w) for w in data["vocab"]],
                [data[f"logp{k}"] for k in range(order)],
                [data[f"backoff{k}"] for k in range(order)],
                [data[f"keys{k}"] for k in range(order)],
            )

    def to_arpa(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            handle.write("\\data\\\n")
            for k in range(self.order):
                handle.write(f"ngram {k + 1}={len(self._logp[k])}\n")
            for k in range(self.order):
                handle.write(f"\n\\{k + 1}-grams:\n")
                last = k == self.order - 1
                for row in range(len(self._logp[k])):
                    key = row if k == 0 else int(self._keys[k][row])
                    ids = []
                    for _ in range(k + 1):
                        key, word = divmod(key, self._size)
                        ids.append(word)
                    words = " ".join(self.vocab[i] for i in reversed(ids))
                    value = self._logp[k][row]
                    line = f"{-99.0 if value <= _MISSING else value / _LN10:.6f}\t{words}"
                    if not last and self._backoff[k][row] != 0.0:
                        line += f"\t{self._backoff[k][row] / _LN10:.6f}"
                    handle.write(line + "\n")
            handle.write("\n\\end\\\n")

    @classmethod
    def from_arpa(cls, path: str | Path, *, vocab: Iterable[str] | None = None) -> NGramLM:
        """Import an ARPA model. ``vocab`` optionally restricts it to a word list."""
        path = Path(path)
        opener = gzip.open if path.suffix == ".gz" else open
        allowed = set(vocab) | {BOS, EOS, UNK} if vocab is not None else None
        sections: dict[int, list[tuple[tuple[str, ...], float, float]]] = {}
        current = 0
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith(("\\data", "ngram ")):
                    continue
                if line.startswith("\\end"):
                    break
                if line.startswith("\\") and line.endswith("-grams:"):
                    current = int(line[1 : line.index("-")])
                    sections[current] = []
                    continue
                if current == 0:
                    continue
                parts = line.split("\t") if "\t" in line else line.split()
                if "\t" in line:
                    words = tuple(parts[1].split())
                    back = float(parts[2]) if len(parts) > 2 else 0.0
                else:
                    words = tuple(parts[1 : 1 + current])
                    back = float(parts[1 + current]) if len(parts) > 1 + current else 0.0
                words = tuple(
                    UNK if w in ("<UNK>", "<unk>") else w.lower() if w not in (BOS, EOS) else w for w in words
                )
                if allowed is not None and any(w not in allowed for w in words):
                    continue
                sections[current].append((words, float(parts[0]) * _LN10, back * _LN10))
        order = max(sections)
        words = [BOS, EOS, UNK, *sorted({w[0] for w, _, _ in sections[1]} - {BOS, EOS, UNK})]
        index = {w: i for i, w in enumerate(words)}
        size = len(words)
        logp = [np.full(size, _MISSING)]
        backoff = [np.zeros(size)]
        for (word,), p, b in sections[1]:
            logp[0][index[word]] = p
            backoff[0][index[word]] = b
        keys: list[np.ndarray] = [np.zeros(0, dtype=np.uint64)]
        for k in range(2, order + 1):
            packed = np.empty(len(sections[k]), dtype=np.uint64)
            probs = np.empty(len(sections[k]))
            backs = np.empty(len(sections[k]))
            for row, (gram, p, b) in enumerate(sections[k]):
                key = 0
                for word in gram:
                    key = key * size + index[word]
                packed[row], probs[row], backs[row] = key, p, b
            order_index = np.argsort(packed, kind="stable")
            keys.append(packed[order_index])
            logp.append(probs[order_index])
            backoff.append(backs[order_index])
        return cls(words, logp, backoff, keys)


# ------------------------------------------------------------------- training
def _discount(counts: np.ndarray) -> float:
    n1 = float(np.sum(counts == 1))
    n2 = float(np.sum(counts == 2))
    if n1 == 0 or n2 == 0:
        return 0.75
    return float(np.clip(n1 / (n1 + 2 * n2), 0.1, 0.95))


def train_ngram(
    sentences: Iterable[Sequence[str]],
    *,
    order: int = 3,
    vocab: Iterable[str] | None = None,
    max_vocab: int | None = None,
    extra_words: Iterable[str] = (),
    min_count: int = 1,
) -> NGramLM:
    """Estimate an interpolated Kneser-Ney model.

    ``vocab`` fixes the vocabulary (e.g. to the words the lexicon can pronounce);
    otherwise the ``max_vocab`` most frequent words are used. Other words map to
    ``<unk>``. ``extra_words`` are added even if unseen (control gestures, names),
    and receive unigram mass through interpolation with the uniform
    distribution. ``min_count`` prunes highest-order n-grams seen fewer times.
    """
    if not 1 <= order <= 4:
        raise ValueError("order must be between 1 and 4")
    corpus = [list(s) for s in sentences if len(s) > 0]
    if not corpus:
        raise ValueError("cannot train a language model on an empty corpus")
    counts = Counter(word for sentence in corpus for word in sentence)
    if vocab is not None:
        chosen = [w for w in dict.fromkeys(vocab) if w not in (BOS, EOS, UNK)]
    else:
        chosen = [w for w, _ in counts.most_common(max_vocab)]
    for word in extra_words:
        if word not in chosen:
            chosen.append(word)
    words = [BOS, EOS, UNK, *sorted(set(chosen))]
    index = {w: i for i, w in enumerate(words)}
    V = len(words)
    if V**order >= 2**63:
        raise ValueError("vocabulary too large for this n-gram order")

    lengths = np.fromiter((len(s) + 2 for s in corpus), dtype=np.int64, count=len(corpus))
    ids = np.empty(int(lengths.sum()), dtype=np.int64)
    sentence_of = np.repeat(np.arange(len(corpus)), lengths)
    cursor = 0
    for sentence in corpus:
        ids[cursor] = BOS_ID
        ids[cursor + 1 : cursor + 1 + len(sentence)] = [index.get(w, UNK_ID) for w in sentence]
        ids[cursor + 1 + len(sentence)] = EOS_ID
        cursor += len(sentence) + 2

    # Raw k-gram types and counts (windows never cross a sentence boundary).
    raw_keys: list[np.ndarray] = []
    raw_counts: list[np.ndarray] = []
    for k in range(1, order + 1):
        n = len(ids) - k + 1
        key = np.zeros(n, dtype=np.int64)
        for offset in range(k):
            key = key * V + ids[offset : offset + n]
        inside = sentence_of[:n] == sentence_of[k - 1 :]
        unique, count = np.unique(key[inside], return_counts=True)
        raw_keys.append(unique)
        raw_counts.append(count.astype(np.float64))

    # Kneser-Ney adjusted counts: below the top order, an n-gram counts once per
    # distinct word that precedes it (n-grams starting with <s> keep raw counts).
    keys: list[np.ndarray] = [None] * order  # type: ignore[list-item]
    adjusted: list[np.ndarray] = [None] * order  # type: ignore[list-item]
    keys[order - 1], adjusted[order - 1] = raw_keys[order - 1], raw_counts[order - 1]
    for k in range(order - 1, 0, -1):  # k-grams from (k+1)-gram types
        suffix, continuation = np.unique(raw_keys[k] % (V**k), return_counts=True)
        table = dict(zip(suffix.tolist(), continuation.tolist()))
        first = raw_keys[k - 1] // (V ** (k - 1))
        starts_with_bos = first == BOS_ID
        for key_value, count in zip(
            raw_keys[k - 1][starts_with_bos].tolist(), raw_counts[k - 1][starts_with_bos].tolist()
        ):
            table[key_value] = count
        ordered = np.array(sorted(table), dtype=np.int64)
        keys[k - 1] = ordered
        adjusted[k - 1] = np.array([table[v] for v in ordered.tolist()], dtype=np.float64)

    # Unigrams: discounted and interpolated with the uniform distribution.
    unigram = np.zeros(V)
    unigram[keys[0]] = adjusted[0]
    unigram[BOS_ID] = 0.0
    d1 = _discount(unigram[unigram > 0])
    total = unigram.sum()
    types = float(np.sum(unigram > 0))
    prob = [(np.maximum(unigram - d1, 0.0) + d1 * types / (V - 1)) / total]
    prob[0][BOS_ID] = 0.0
    backoff = [np.zeros(V)]
    table_keys: list[np.ndarray] = [np.zeros(0, dtype=np.uint64)]
    lower_keys = np.arange(V, dtype=np.int64)
    lower_prob = prob[0]

    for k in range(2, order + 1):
        key, count = keys[k - 1], adjusted[k - 1]
        context = key // V
        _, starts, types_per_context = np.unique(context, return_index=True, return_counts=True)
        mass = np.repeat(np.add.reduceat(count, starts), types_per_context)
        spread = np.repeat(types_per_context.astype(np.float64), types_per_context)
        discount = _discount(count)
        suffix_row = np.searchsorted(lower_keys, key % (V ** (k - 1)))
        lower = lower_prob[suffix_row]
        p = np.maximum(count - discount, 0.0) / mass + discount * spread / mass * lower

        if k == order and min_count > 1:
            keep = raw_counts[k - 1] >= min_count
            key, p, lower, context = key[keep], p[keep], lower[keep], context[keep]
        # Back-off weights from the mass actually left over, so pruning stays normalised.
        contexts, starts = np.unique(context, return_index=True)
        left = 1.0 - np.add.reduceat(p, starts) if len(p) else np.zeros(0)
        left_lower = 1.0 - np.add.reduceat(lower, starts) if len(p) else np.zeros(0)
        weight = np.where(left_lower > 1e-12, np.maximum(left, 1e-12) / np.maximum(left_lower, 1e-12), 1.0)
        context_backoff = np.zeros(len(lower_keys))
        context_backoff[np.searchsorted(lower_keys, contexts)] = np.log(weight)
        backoff[k - 2] = context_backoff

        prob.append(p)
        backoff.append(np.zeros(len(p)))
        table_keys.append(key.astype(np.uint64))
        lower_keys, lower_prob = key, p

    with np.errstate(divide="ignore"):
        logp = [np.where(p > 0, np.log(np.maximum(p, 1e-300)), _MISSING) for p in prob]
    return NGramLM(words, logp, backoff, table_keys)
