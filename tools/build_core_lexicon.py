#!/usr/bin/env python3
"""Rebuild the bundled core lexicon (``src/ssj/phonology/data/core.dict``).

The core lexicon is the subset of CMUdict covering the most frequent words of a
text corpus plus every word used by the built-in recording prompts, so that the
package can pronounce and decode common English without any downloads.

    python tools/build_core_lexicon.py \
        --cmudict assets/lexicon/cmudict.dict \
        --corpus assets/corpus/eng_sentences.tsv --size 8000

The corpus may be plain text (one sentence per line) or a Tatoeba TSV export.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from ssj.phonology import Inventory, Lexicon, normalize_text, number_to_words

ROOT = Path(__file__).resolve().parents[1]
HEADER = """\
;;; Core pronunciation lexicon for silent-speech-jaw.
;;; Derived from the CMU Pronouncing Dictionary (stress markers removed, restricted
;;; to frequent words). Copyright (C) 1993-2015 Carnegie Mellon University.
;;; Redistributed under the CMUdict licence; see THIRD_PARTY_NOTICES.md.
;;; Regenerate with tools/build_core_lexicon.py - do not edit by hand.
"""


def read_sentences(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            yield line.rsplit("\t", 1)[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cmudict", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--size", type=int, default=8000, help="number of frequent words to keep")
    parser.add_argument("--max-variants", type=int, default=3)
    parser.add_argument("--out", type=Path, default=ROOT / "src/ssj/phonology/data/core.dict")
    args = parser.parse_args()

    inventory = Inventory.default()
    full = Lexicon.load(args.cmudict, inventory=inventory)
    counts: Counter[str] = Counter()
    for sentence in read_sentences(args.corpus):
        counts.update(normalize_text(sentence))

    ranked = [w for w, _ in counts.most_common() if w in full]
    chosen = set(ranked[: args.size])

    required: set[str] = set()
    for sentence in read_sentences(ROOT / "src/ssj/collect/data/sentences.txt"):
        required.update(normalize_text(sentence))
    for n in [*range(0, 100), 100, 1000, 1_000_000]:
        required.update(number_to_words(n))
    required.update(
        "first second third fourth fifth sixth seventh eighth ninth tenth percent and plus at equals minus".split()
    )
    unknown = sorted(w for w in required if w not in full)
    if unknown:
        raise SystemExit(f"prompt words missing from CMUdict: {unknown}")
    chosen |= required

    total = sum(counts.values())
    covered = sum(c for w, c in counts.items() if w in chosen)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        handle.write(HEADER)
        for word in sorted(chosen):
            for i, pron in enumerate(full.prons(word)[: args.max_variants]):
                key = word if i == 0 else f"{word}({i + 1})"
                handle.write(f"{key} {' '.join(pron)}\n")
    print(f"corpus tokens: {total:,}; distinct words: {len(counts):,}")
    print(f"core lexicon: {len(chosen):,} words, token coverage {covered / total:.1%}")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
