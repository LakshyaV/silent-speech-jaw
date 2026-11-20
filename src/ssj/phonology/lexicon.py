"""Pronunciation lexicon in CMUdict format.

The lexicon is what turns phoneme recognition into open-vocabulary word
recognition: any word with a pronunciation can be decoded without ever having
been recorded. A compact core lexicon ships with the package; the full CMUdict
(about 135k words) is fetched with ``ssj assets fetch lexicon``.
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Iterable, Iterator, Sequence
from importlib import resources
from pathlib import Path

from .inventory import Inventory
from .text import CONTROL_WORDS, normalize_text

Pron = tuple[str, ...]

_WORD = re.compile(r"^[a-z][a-z']*$")
_VARIANT = re.compile(r"\(\d+\)$")
_STRESS = re.compile(r"\d")


class OOVError(KeyError):
    """Raised when text contains words the lexicon cannot pronounce."""

    def __init__(self, words: Sequence[str]):
        self.words = tuple(words)
        super().__init__(f"no pronunciation for: {', '.join(self.words)}")


def _open_text(path: Path) -> Iterator[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="latin-1") as handle:  # CMUdict 0.7b is latin-1
        yield from handle


class Lexicon:
    """Mapping ``word -> [pronunciation, ...]`` with stress-free ARPAbet symbols."""

    def __init__(self, entries: dict[str, list[Pron]] | None = None):
        self._entries: dict[str, list[Pron]] = {}
        for word, prons in (entries or {}).items():
            for pron in prons:
                self.add(word, pron)

    # --------------------------------------------------------------- parsing
    @staticmethod
    def parse_lines(lines: Iterable[str]) -> Iterator[tuple[str, Pron]]:
        for line in lines:
            line = line.split(" #", 1)[0].strip()
            if not line or line.startswith(";;;"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            word = _VARIANT.sub("", parts[0].lower())
            if word not in CONTROL_WORDS and not _WORD.match(word):
                continue
            yield word, tuple(_STRESS.sub("", p).upper() for p in parts[1:])

    @classmethod
    def load(
        cls, path: str | Path, *, inventory: Inventory | None = None, keep: Iterable[str] | None = None
    ) -> Lexicon:
        """Load a CMUdict-style file (optionally gzipped).

        ``keep`` restricts the lexicon to a vocabulary; ``inventory`` drops
        pronunciations that use symbols outside the model's token set.
        """
        wanted = set(keep) if keep is not None else None
        lexicon = cls()
        for word, pron in cls.parse_lines(_open_text(Path(path))):
            if wanted is not None and word not in wanted:
                continue
            if inventory is not None and any(p not in inventory for p in pron):
                continue
            lexicon.add(word, pron)
        return lexicon

    @classmethod
    def core(cls, inventory: Inventory | None = None) -> Lexicon:
        """The bundled core lexicon (common English words plus control gestures)."""
        data = resources.files("ssj.phonology").joinpath("data/core.dict").read_text("utf-8")
        lexicon = cls()
        for word, pron in cls.parse_lines(data.splitlines()):
            if inventory is None or all(p in inventory for p in pron):
                lexicon.add(word, pron)
        lexicon.add_control_words(inventory)
        return lexicon

    # -------------------------------------------------------------- mutation
    def add(self, word: str, pron: Sequence[str]) -> None:
        pron = tuple(pron)
        if not pron:
            raise ValueError(f"empty pronunciation for {word!r}")
        prons = self._entries.setdefault(word, [])
        if pron not in prons:
            prons.append(pron)

    def add_control_words(self, inventory: Inventory | None = None) -> None:
        for word, pron in CONTROL_WORDS.items():
            if inventory is None or all(p in inventory for p in pron):
                self.add(word, pron)

    def merge(self, other: Lexicon) -> Lexicon:
        for word, prons in other._entries.items():
            for pron in prons:
                self.add(word, pron)
        return self

    def subset(self, words: Iterable[str]) -> Lexicon:
        wanted = set(words)
        return Lexicon({w: list(p) for w, p in self._entries.items() if w in wanted})

    # ---------------------------------------------------------------- access
    def __contains__(self, word: object) -> bool:
        return word in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    @property
    def words(self) -> list[str]:
        return list(self._entries)

    def prons(self, word: str) -> list[Pron]:
        return self._entries[word]

    def items(self) -> Iterator[tuple[str, list[Pron]]]:
        return iter(self._entries.items())

    def n_pronunciations(self) -> int:
        return sum(len(p) for p in self._entries.values())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for word in sorted(self._entries):
                for i, pron in enumerate(self._entries[word]):
                    key = word if i == 0 else f"{word}({i + 1})"
                    handle.write(f"{key} {' '.join(pron)}\n")

    # ------------------------------------------------------------ text -> phonemes
    def missing(self, words: Iterable[str]) -> list[str]:
        seen: list[str] = []
        for word in words:
            if word not in self._entries and word not in seen:
                seen.append(word)
        return seen

    def phonemize_words(self, words: Sequence[str], variants: Sequence[int] | None = None) -> list[Pron]:
        """Pronunciation of each word; ``variants[i]`` selects among alternatives."""
        missing = self.missing(words)
        if missing:
            raise OOVError(missing)
        out: list[Pron] = []
        for i, word in enumerate(words):
            prons = self._entries[word]
            out.append(prons[(variants[i] if variants else 0) % len(prons)])
        return out

    def phonemize(self, text: str | Sequence[str]) -> list[str]:
        """Flat phoneme sequence for a sentence, using each word's first pronunciation."""
        words = normalize_text(text) if isinstance(text, str) else list(text)
        return [p for pron in self.phonemize_words(words) for p in pron]

    def homophene_sets(self, inventory: Inventory) -> dict[tuple[int, ...], list[str]]:
        """Group words whose pronunciations collapse to the same class sequence.

        These are the words the sensor alone cannot tell apart (pat/bat/mat);
        the size of these sets is the ambiguity the language models must resolve.
        """
        groups: dict[tuple[int, ...], list[str]] = {}
        for word, prons in self._entries.items():
            for pron in prons:
                if any(p not in inventory for p in pron):
                    continue
                key = tuple(inventory.to_class_ids(inventory.encode(pron)))
                bucket = groups.setdefault(key, [])
                if word not in bucket:
                    bucket.append(word)
        return groups
