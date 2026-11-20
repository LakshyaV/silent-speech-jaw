"""Phoneme inventory with a two-level (phoneme / articulatory class) token space.

Token id 0 is the CTC blank at both levels. Phoneme ids run 1..V and class ids
1..K. The mapping phoneme -> class is many-to-one and is what lets the rest of
the system reason at the resolution the sensor actually has.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..config import fingerprint

BLANK = "<blank>"
BLANK_ID = 0


class InventoryError(ValueError):
    pass


@dataclass(frozen=True)
class Inventory:
    phonemes: tuple[str, ...]
    classes: tuple[str, ...]
    class_of: tuple[int, ...]  # 1-based class id for each phoneme
    kinds: tuple[str, ...]  # "vowel" | "consonant" | "gesture"

    def __post_init__(self) -> None:
        if len(set(self.phonemes)) != len(self.phonemes):
            raise InventoryError("duplicate phoneme symbols")
        if len(set(self.classes)) != len(self.classes):
            raise InventoryError("duplicate class names")
        if not (len(self.phonemes) == len(self.class_of) == len(self.kinds)):
            raise InventoryError("phonemes, class_of and kinds must have equal length")
        if any(not 1 <= c <= len(self.classes) for c in self.class_of):
            raise InventoryError("class id out of range")
        unused = set(range(1, len(self.classes) + 1)) - set(self.class_of)
        if unused:
            names = [self.classes[c - 1] for c in sorted(unused)]
            raise InventoryError(f"classes without members: {names}")

    # ------------------------------------------------------------------ sizes
    @property
    def size(self) -> int:
        """Number of fine output units including blank."""
        return len(self.phonemes) + 1

    @property
    def n_classes(self) -> int:
        """Number of class-level output units including blank."""
        return len(self.classes) + 1

    # --------------------------------------------------------------- encoding
    @cached_property
    def _index(self) -> dict[str, int]:
        return {symbol: i + 1 for i, symbol in enumerate(self.phonemes)}

    def __contains__(self, symbol: object) -> bool:
        return symbol in self._index

    def token_id(self, symbol: str) -> int:
        try:
            return self._index[symbol]
        except KeyError:
            raise InventoryError(f"unknown phoneme {symbol!r}") from None

    def symbol(self, token_id: int) -> str:
        if token_id == BLANK_ID:
            return BLANK
        return self.phonemes[token_id - 1]

    def encode(self, symbols: Iterable[str]) -> list[int]:
        return [self.token_id(s) for s in symbols]

    def decode(self, token_ids: Iterable[int]) -> list[str]:
        return [self.symbol(int(i)) for i in token_ids]

    # ---------------------------------------------------------------- classes
    @cached_property
    def class_ids(self) -> np.ndarray:
        """``class_ids[token_id] -> class_id`` with blank mapped to blank."""
        return np.asarray((BLANK_ID, *self.class_of), dtype=np.int64)

    def class_name(self, class_id: int) -> str:
        return BLANK if class_id == BLANK_ID else self.classes[class_id - 1]

    def to_class_ids(self, token_ids: Iterable[int]) -> list[int]:
        table = self.class_ids
        return [int(table[int(i)]) for i in token_ids]

    def members(self, class_name: str) -> tuple[str, ...]:
        cid = self.classes.index(class_name) + 1
        return tuple(p for p, c in zip(self.phonemes, self.class_of) if c == cid)

    @cached_property
    def membership(self) -> np.ndarray:
        """Boolean matrix ``[size, n_classes]``; ``membership[v, k]`` iff token v is in class k."""
        matrix = np.zeros((self.size, self.n_classes), dtype=bool)
        matrix[np.arange(self.size), self.class_ids] = True
        return matrix

    def is_vowel(self, symbol: str) -> bool:
        return self.kinds[self.token_id(symbol) - 1] == "vowel"

    @cached_property
    def gestures(self) -> tuple[str, ...]:
        return tuple(p for p, k in zip(self.phonemes, self.kinds) if k == "gesture")

    # ----------------------------------------------------------- construction
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Inventory:
        class_names = tuple(data["classes"])
        lookup = {name: i + 1 for i, name in enumerate(class_names)}
        phonemes, class_of, kinds = [], [], []
        for entry in data["phonemes"]:
            name = entry["class"]
            if name not in lookup:
                raise InventoryError(f"{entry['symbol']}: unknown class {name!r}")
            phonemes.append(str(entry["symbol"]))
            class_of.append(lookup[name])
            kinds.append(str(entry.get("kind", "consonant")))
        return cls(tuple(phonemes), class_names, tuple(class_of), tuple(kinds))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "classes": list(self.classes),
            "phonemes": [
                {"symbol": p, "class": self.classes[c - 1], "kind": k}
                for p, c, k in zip(self.phonemes, self.class_of, self.kinds)
            ],
        }

    @classmethod
    def from_yaml(cls, path: str | Path) -> Inventory:
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle))

    @classmethod
    def default(cls) -> Inventory:
        """ARPAbet-39 plus control gestures with the built-in articulatory classes."""
        text = resources.files("ssj.phonology").joinpath("data/arpabet.yaml").read_text("utf-8")
        return cls.from_dict(yaml.safe_load(text))

    def with_class_map(self, mapping: Mapping[str, str]) -> Inventory:
        """Return a copy whose phoneme -> class assignment is replaced by ``mapping``.

        Used by inventory refinement: the phoneme set (and therefore the model's
        output layer) is unchanged, only the grouping moves.
        """
        missing = [p for p in self.phonemes if p not in mapping]
        if missing:
            raise InventoryError(f"class map is missing phonemes {missing}")
        names: list[str] = []
        for p in self.phonemes:
            if mapping[p] not in names:
                names.append(mapping[p])
        lookup = {name: i + 1 for i, name in enumerate(names)}
        return Inventory(self.phonemes, tuple(names), tuple(lookup[mapping[p]] for p in self.phonemes), self.kinds)

    def without(self, symbols: Sequence[str]) -> Inventory:
        """Return a copy with some tokens removed (e.g. to train without gestures)."""
        drop = set(symbols)
        entries = [e for e in self.to_dict()["phonemes"] if e["symbol"] not in drop]
        used = {e["class"] for e in entries}
        return Inventory.from_dict({"classes": [c for c in self.classes if c in used], "phonemes": entries})

    @cached_property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())
