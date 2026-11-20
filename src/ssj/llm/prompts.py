"""Prompt templates for the LLM stage.

Scoring uses a plain-text frame rather than a chat template: every candidate is
scored as the next line of a short transcript, which is the setting in which a
language model's probabilities are most meaningful and which works identically
for base and instruction-tuned models.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..decode.lattice import Slot

SCORING_PREAMBLE = "Transcript of short, everyday things one person said, one sentence per line.\n"

SYSTEM = (
    "You reconstruct what a person said without sound. A jaw motion sensor detected their mouth "
    "movements, which cannot tell apart similar-looking sounds such as P, B and M. "
    "Reply with the single most natural English sentence and nothing else."
)


def scoring_context(history: Sequence[str] = (), hint: str = "", max_history: int = 4) -> str:
    """Text that precedes a candidate sentence when it is scored."""
    lines = [SCORING_PREAMBLE]
    if hint:
        lines.append(f"Situation: {hint.strip()}\n")
    lines.extend(f"{sentence}\n" for sentence in list(history)[-max_history:])
    return "".join(lines)


def _context_lines(history: Sequence[str], hint: str, max_history: int = 3) -> list[str]:
    lines = []
    if hint:
        lines.append(f"Situation: {hint.strip()}")
    recent = list(history)[-max_history:]
    if recent:
        lines.append("Previously said: " + " ".join(recent))
    return lines


def lattice_prompt(
    slots: Sequence[Slot], history: Sequence[str] = (), hint: str = "", limit: int = 6
) -> tuple[str, str]:
    """Prompt for grammar-constrained selection of one path through the lattice."""
    lines = _context_lines(history, hint)
    lines.append("Candidate words for each position, most likely first (positions in brackets may be skipped):")
    for number, slot in enumerate(slots, start=1):
        words = " / ".join(slot.choices()[:limit])
        lines.append(f"{number}. {'[' + words + ']' if slot.optional else words}")
    lines.append("Choose one word per position so that the result is a natural sentence.")
    return SYSTEM, "\n".join(lines)


def proposal_prompt(
    phonemes: Sequence[str], guesses: Sequence[str], history: Sequence[str] = (), hint: str = ""
) -> tuple[str, str]:
    """Prompt for an unconstrained proposal from the detected phonemes and the best guesses."""
    lines = _context_lines(history, hint)
    lines.append("Detected sounds (ARPAbet, approximate): " + " ".join(phonemes))
    if guesses:
        lines.append("Best guesses so far:")
        lines.extend(f"- {guess}" for guess in guesses)
    lines.append("Write the sentence the person most likely said.")
    return SYSTEM, "\n".join(lines)
