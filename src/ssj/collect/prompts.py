"""Prompt design for recording sessions.

Because recognition is phoneme-based, what matters in a recording session is
not *which words* are spoken but how well the phonemes and their transitions are
covered. Prompts are therefore chosen by greedy set cover over diphones (pairs
of adjacent phonemes) from any pool of sentences, preceded by isolated-word
drills that present every phoneme at the start, middle and end of a word.
"""

from __future__ import annotations

import heapq
import json
from collections import Counter
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from importlib import resources
from itertools import pairwise
from pathlib import Path

import numpy as np

from ..phonology import CONTROL_WORDS, Inventory, Lexicon, normalize_text


@dataclass
class Prompt:
    prompt_id: str
    text: str
    kind: str = "sentence"  # word | sentence | gesture
    phonemes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"prompt_id": self.prompt_id, "text": self.text, "kind": self.kind, "phonemes": self.phonemes}

    @classmethod
    def from_dict(cls, data: dict) -> Prompt:
        return cls(
            str(data["prompt_id"]),
            str(data["text"]),
            str(data.get("kind", "sentence")),
            list(data.get("phonemes") or []),
        )


def builtin_sentences() -> list[str]:
    text = resources.files("ssj.collect").joinpath("data/sentences.txt").read_text("utf-8")
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def read_text_pool(path: str | Path, limit: int | None = None) -> list[str]:
    """Read sentences from plain text (one per line) or a Tatoeba TSV export."""
    out: list[str] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line.rsplit("\t", 1)[-1])
            if limit is not None and len(out) >= limit:
                break
    return out


def diphones(phonemes: Sequence[str]) -> list[tuple[str, str]]:
    padded = ["#", *phonemes, "#"]
    return list(pairwise(padded))


def positional_units(phonemes: Sequence[str]) -> list[tuple[str, str]]:
    """(phoneme, position) units for word drills."""
    units = []
    for i, phoneme in enumerate(phonemes):
        position = "initial" if i == 0 else "final" if i == len(phonemes) - 1 else "medial"
        units.append((phoneme, position))
    return units


def greedy_cover(
    candidates: Sequence[Sequence[Hashable]],
    n: int,
    *,
    target: int = 1,
    length_power: float = 0.6,
) -> list[int]:
    """Pick ``n`` candidates maximising unit coverage (lazy greedy set cover).

    Each unit is worth covering ``target`` times; the gain of a candidate is the
    number of still-needed unit occurrences it supplies, discounted by its length
    so that long candidates do not win by sheer size. Once everything has been
    covered ``target`` times the target is raised, so the selection keeps
    balancing counts instead of degenerating to arbitrary picks.
    """
    counts: Counter = Counter()
    lengths = [max(len(c), 1) ** length_power for c in candidates]
    unit_sets = [Counter(c) for c in candidates]

    def gain(index: int, goal: int) -> float:
        need = 0
        for unit, occurrences in unit_sets[index].items():
            need += min(occurrences, max(goal - counts[unit], 0))
        return need / lengths[index]

    chosen: list[int] = []
    remaining = set(range(len(candidates)))
    goal = target
    while len(chosen) < n and remaining:
        # One round per coverage goal. Gains only shrink within a round
        # (submodularity), so a popped score that is still the best is exact.
        heap = [(-gain(i, goal), i) for i in remaining]
        heapq.heapify(heap)
        progressed = False
        while heap and len(chosen) < n:
            _, index = heapq.heappop(heap)
            fresh = gain(index, goal)
            if fresh <= 0:
                continue
            if heap and fresh < -heap[0][0] - 1e-12:
                heapq.heappush(heap, (-fresh, index))
                continue
            chosen.append(index)
            remaining.discard(index)
            counts.update(unit_sets[index])
            progressed = True
        if not progressed and goal > max(counts.values(), default=0):
            break  # the remaining candidates carry no units at all
        goal += 1
    return chosen


def _usable(words: list[str], lexicon: Lexicon, min_words: int, max_words: int) -> bool:
    return min_words <= len(words) <= max_words and not lexicon.missing(words)


def word_drills(lexicon: Lexicon, inventory: Inventory, n: int, *, max_phonemes: int = 5) -> list[str]:
    """Isolated words that present every phoneme in initial, medial and final position."""
    words = [w for w in lexicon.words if w not in CONTROL_WORDS and 2 <= len(lexicon.prons(w)[0]) <= max_phonemes]
    units = [positional_units(lexicon.prons(w)[0]) for w in words]
    return [words[i] for i in greedy_cover(units, n, length_power=0.3)]


def build_prompt_set(
    lexicon: Lexicon,
    inventory: Inventory,
    *,
    n_words: int = 60,
    n_sentences: int = 120,
    pool: Iterable[str] | None = None,
    min_words: int = 2,
    max_words: int = 9,
    seed: int = 0,
    clench_every: int = 0,
) -> list[Prompt]:
    """Assemble one session's prompt list: word drills first, then sentences.

    ``clench_every`` > 0 appends the ``<clench>`` control gesture to every n-th
    sentence so the recogniser learns the gesture in context.
    """
    rng = np.random.default_rng(seed)
    prompts: list[Prompt] = []

    drills = word_drills(lexicon, inventory, n_words)
    for index in rng.permutation(len(drills)):
        word = drills[int(index)]
        prompts.append(Prompt(f"w{len(prompts):04d}", word, "word", list(lexicon.prons(word)[0])))

    sentences: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for raw in pool if pool is not None else builtin_sentences():
        words = normalize_text(raw)
        key = tuple(words)
        if key in seen or not _usable(words, lexicon, min_words, max_words):
            continue
        seen.add(key)
        sentences.append(words)
    order = rng.permutation(len(sentences))  # break ties differently for every seed
    sentences = [sentences[int(i)] for i in order]
    units = [diphones(lexicon.phonemize(words)) for words in sentences]
    picked = greedy_cover(units, n_sentences)
    for position, index in enumerate(rng.permutation(len(picked))):
        words = list(sentences[picked[int(index)]])
        if clench_every and (position + 1) % clench_every == 0:
            words.append("<clench>")
        prompts.append(Prompt(f"s{position:04d}", " ".join(words), "sentence", lexicon.phonemize(words)))
    return prompts


def coverage_report(prompts: Sequence[Prompt], inventory: Inventory) -> dict:
    """How well a prompt set covers phonemes and diphones."""
    phoneme_counts: Counter = Counter()
    diphone_counts: Counter = Counter()
    for prompt in prompts:
        phoneme_counts.update(prompt.phonemes)
        diphone_counts.update(d for d in diphones(prompt.phonemes) if "#" not in d)
    speech = [p for p, k in zip(inventory.phonemes, inventory.kinds) if k != "gesture"]
    missing = [p for p in speech if phoneme_counts[p] == 0]
    return {
        "prompts": len(prompts),
        "phoneme_tokens": int(sum(phoneme_counts.values())),
        "phonemes_covered": len(speech) - len(missing),
        "phonemes_total": len(speech),
        "missing_phonemes": missing,
        "min_phoneme_count": int(min((phoneme_counts[p] for p in speech), default=0)),
        "distinct_diphones": len(diphone_counts),
        "rarest_phonemes": [
            (p, int(c)) for p, c in sorted(((p, phoneme_counts[p]) for p in speech), key=lambda x: x[1])[:5]
        ],
    }


def save_prompts(path: str | Path, prompts: Sequence[Prompt]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            handle.write(json.dumps(prompt.to_dict(), ensure_ascii=False) + "\n")


def load_prompts(path: str | Path) -> list[Prompt]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [Prompt.from_dict(json.loads(line)) for line in handle if line.strip()]
