"""Text normalisation shared by prompts, language models and scoring.

Everything that touches words goes through :func:`normalize_text`, so prompts,
n-gram training text, decoder output and reference transcripts all live in the
same token space: lower-case ASCII words made of letters and apostrophes.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence

_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_ORDINAL_SUFFIX = re.compile(r"^(\d+)(st|nd|rd|th)$")
_ORDINALS = {
    "one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
    "nine": "ninth", "twelve": "twelfth",
}  # fmt: skip
_TOKEN = re.compile(r"[a-z0-9']+")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_QUOTES = str.maketrans({"’": "'", "‘": "'", "`": "'", "“": '"', "”": '"', "–": " ", "—": " ", "-": " "})
_SYMBOLS = {"&": " and ", "%": " percent ", "+": " plus ", "@": " at ", "=": " equals "}

#: Pseudo-words for control gestures. They are ordinary lexicon entries whose
#: pronunciation is the gesture token, so the decoder emits them like any word.
CONTROL_WORDS = {"<clench>": ("CLENCH",)}


def number_to_words(n: int) -> list[str]:
    """Spell a non-negative integer below one billion as English words."""
    if n < 0:
        return ["minus", *number_to_words(-n)]
    if n < 20:
        return [_ONES[n]]
    if n < 100:
        tens, ones = divmod(n, 10)
        return [_TENS[tens]] + ([_ONES[ones]] if ones else [])
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return [_ONES[hundreds], "hundred"] + (number_to_words(rest) if rest else [])
    for value, name in ((1_000_000, "million"), (1_000, "thousand")):
        if n >= value:
            head, rest = divmod(n, value)
            return [*number_to_words(head), name] + (number_to_words(rest) if rest else [])
    raise ValueError(n)


def _ordinal(n: int) -> list[str]:
    words = number_to_words(n)
    last = words[-1]
    if last in _ORDINALS:
        words[-1] = _ORDINALS[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return words


def normalize_text(text: str) -> list[str]:
    """Split free text into normalised word tokens.

    >>> normalize_text("It's 21 o'clock — I'll take the 3rd one!")
    ["it's", 'twenty', 'one', "o'clock", "i'll", 'take', 'the', 'third', 'one']
    """
    controls: list[tuple[int, str]] = []
    lowered = unicodedata.normalize("NFKC", text).lower().translate(_QUOTES)
    for word in CONTROL_WORDS:
        while word in lowered:
            index = lowered.index(word)
            marker = f" ctrlmark{len(controls)}x "
            controls.append((len(controls), word))
            lowered = lowered[:index] + marker + lowered[index + len(word) :]
    lowered = _THOUSANDS.sub("", lowered)
    for symbol, spoken in _SYMBOLS.items():
        lowered = lowered.replace(symbol, spoken)
    lowered = unicodedata.normalize("NFKD", lowered).encode("ascii", "ignore").decode("ascii")

    words: list[str] = []
    for token in _TOKEN.findall(lowered):
        token = token.strip("'")
        if not token:
            continue
        if token.startswith("ctrlmark") and token.endswith("x") and token[8:-1].isdigit():
            words.append(controls[int(token[8:-1])][1])
            continue
        ordinal = _ORDINAL_SUFFIX.match(token)
        if ordinal and len(ordinal.group(1)) <= 9:
            words.extend(_ordinal(int(ordinal.group(1))))
        elif token.isdigit():
            if len(token) <= 9:
                words.extend(number_to_words(int(token)))
            else:  # long digit strings are read digit by digit
                words.extend(_ONES[int(d)] for d in token)
        elif any(ch.isdigit() for ch in token):
            for part in re.findall(r"\d+|[a-z']+", token):
                words.extend(normalize_text(part))
        else:
            words.append(token)
    return words


_CAPITALISE = {"i": "I", "i'm": "I'm", "i'll": "I'll", "i've": "I've", "i'd": "I'd"}


def render_sentence(words: Sequence[str], *, punctuate: bool = True) -> str:
    """Turn decoder words into display text (and into the form the LLM scores).

    Language models assign very different probabilities to ``hello world`` and
    ``Hello world.``; rendering every candidate the same way keeps the comparison
    between candidates fair and puts them in the register the LLM models best.
    """
    visible = [_CAPITALISE.get(w, w) for w in words if w not in CONTROL_WORDS]
    if not visible:
        return ""
    visible[0] = visible[0][0].upper() + visible[0][1:]
    text = " ".join(visible)
    return text + "." if punctuate else text


def join_words(words: Iterable[str]) -> str:
    return " ".join(words)
