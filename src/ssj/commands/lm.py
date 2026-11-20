"""Build or inspect the n-gram language model used by the first decoding pass.

    ssj lm build                                   # from the fetched corpus and lexicon
    ssj lm build --text notes.txt --vocab-size 20000 --out assets/lm/mine.npz
    ssj lm info assets/lm/en.npz

The vocabulary is the ``--vocab-size`` most frequent corpus words that the
lexicon can pronounce: a word the decoder cannot spell out in phonemes is of no
use to it. Adding your own text (messages, notes, domain phrases) to the corpus
is the cheapest way to adapt the system to how you actually talk.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

from .. import assets

NAME = "lm"
HELP = "build the n-gram language model from text"
ORDER = 11


def configure(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build", help="train an n-gram model")
    build.add_argument(
        "--text",
        type=Path,
        action="append",
        default=[],
        help="text file, one sentence per line (repeatable; default: the fetched corpus)",
    )
    build.add_argument(
        "--lexicon",
        type=Path,
        default=None,
        help="CMUdict-format lexicon (default: fetched CMUdict, else the bundled core lexicon)",
    )
    build.add_argument("--out", type=Path, default=None, help="output .npz (default: <assets>/lm/en.npz)")
    build.add_argument("--order", type=int, default=3)
    build.add_argument("--vocab-size", type=int, default=30000)
    build.add_argument("--min-count", type=int, default=2, help="prune highest-order n-grams seen fewer times")
    build.add_argument("--max-sentences", type=int, default=None)
    build.add_argument("--arpa", type=Path, default=None, help="also write an ARPA file")
    build.add_argument("--dir", default=None, help="assets directory")
    info = sub.add_parser("info", help="describe a model")
    info.add_argument("path", type=Path)
    info.add_argument("--sentence", default=None, help="score this sentence")


def run(args: argparse.Namespace) -> int:
    from ..collect.prompts import read_text_pool
    from ..decode.ngram import NGramLM, train_ngram
    from ..phonology import CONTROL_WORDS, Inventory, Lexicon, normalize_text

    if args.action == "info":
        lm = NGramLM.load(args.path) if args.path.suffix == ".npz" else NGramLM.from_arpa(args.path)
        print(f"order {lm.order}, vocabulary {len(lm.vocab):,}, n-grams {lm.n_grams()}")
        if args.sentence:
            words = normalize_text(args.sentence)
            print(f"log P({' '.join(words)}) = {lm.score(words):.3f}")
        return 0

    root = assets.assets_dir(args.dir)
    inventory = Inventory.default()
    lexicon_file = args.lexicon or (assets.lexicon_path(root) if assets.lexicon_path(root).exists() else None)
    lexicon = Lexicon.load(lexicon_file, inventory=inventory) if lexicon_file else Lexicon.core(inventory)
    texts = args.text or ([assets.corpus_path(root)] if assets.corpus_path(root).exists() else [])
    if not texts:
        raise SystemExit("no text given and no fetched corpus found: run `ssj assets fetch corpus` or pass --text")

    started = time.time()
    sentences: list[list[str]] = []
    for path in texts:
        for line in read_text_pool(path, args.max_sentences):
            words = normalize_text(line)
            if words:
                sentences.append(words)
    counts = Counter(word for sentence in sentences for word in sentence)
    vocabulary = [word for word, _ in counts.most_common() if word in lexicon][: args.vocab_size]
    covered = sum(counts[w] for w in vocabulary) / max(sum(counts.values()), 1)
    print(
        f"{len(sentences):,} sentences, {sum(counts.values()):,} words; vocabulary {len(vocabulary):,} covers {covered:.1%} of the text"
    )
    lm = train_ngram(sentences, order=args.order, vocab=vocabulary, extra_words=CONTROL_WORDS, min_count=args.min_count)
    out = args.out or assets.lm_path(root)
    lm.save(out)
    print(
        f"order {lm.order}, n-grams {lm.n_grams()} -> {out} ({out.stat().st_size / 1e6:.1f} MB, {time.time() - started:.0f} s)"
    )
    if args.arpa:
        lm.to_arpa(args.arpa)
        print(f"ARPA -> {args.arpa}")
    return 0
