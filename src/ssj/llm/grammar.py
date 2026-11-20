"""Compile a confusion network into a GBNF grammar for constrained generation.

Given the slots ``[{hello, yellow}, {world, word}, {(now)}]`` the grammar is::

    root ::= s0 s1 s2
    s0 ::= " hello" | " yellow"
    s1 ::= " world" | " word"
    s2 ::= (" now")?

Under this grammar the LLM cannot hallucinate: every sentence it can produce is
a path through the lattice, i.e. a word sequence the sensor evidence supports.
What it contributes is the choice among those paths.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..decode.lattice import Slot


def _literal(word: str) -> str:
    escaped = word.replace("\\", "\\\\").replace('"', '\\"')
    return f'" {escaped}"'


def gbnf_from_slots(slots: Sequence[Slot], max_alternatives: int = 6) -> str:
    """GBNF grammar accepting exactly the paths of a confusion network."""
    rules: list[str] = []
    names: list[str] = []
    for index, slot in enumerate(slots):
        choices = slot.choices()[:max_alternatives]
        if not choices:
            continue
        body = " | ".join(_literal(word) for word in choices)
        if slot.optional:
            body = f"({body})?"
        names.append(f"s{index}")
        rules.append(f"s{index} ::= {body}")
    if not names:
        return 'root ::= ""'
    return "\n".join([f"root ::= {' '.join(names)}", *rules]) + "\n"


def n_paths(slots: Sequence[Slot], max_alternatives: int = 6) -> int:
    """Number of distinct sentences the grammar admits."""
    total = 1
    for slot in slots:
        total *= max(len(slot.choices()[:max_alternatives]) + (1 if slot.optional else 0), 1)
    return total
