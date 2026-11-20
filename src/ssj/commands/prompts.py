"""Design the prompt list for a recording session.

    ssj prompts build --out prompts/session01.jsonl
    ssj prompts build --pool assets/corpus/sentences.txt --sentences 200 --seed 3 --out prompts/s03.jsonl

Prompts are selected for phoneme and diphone coverage, not for their meaning:
word drills that place every phoneme at the start, middle and end of a word,
followed by sentences chosen by greedy set cover. A different ``--seed`` gives
a different, equally well-covered list for the next session.
"""

from __future__ import annotations

import argparse
from pathlib import Path

NAME = "prompts"
HELP = "select phonetically balanced recording prompts"
ORDER = 20


def configure(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build", help="create a prompt list")
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--words", type=int, default=60, help="isolated-word drills")
    build.add_argument("--sentences", type=int, default=120)
    build.add_argument("--pool", type=Path, default=None, help="sentence pool (default: the built-in sentences)")
    build.add_argument("--pool-limit", type=int, default=200000)
    build.add_argument("--lexicon", type=Path, default=None)
    build.add_argument("--max-words", type=int, default=9)
    build.add_argument("--clench-every", type=int, default=0, help="append the <clench> gesture to every n-th sentence")
    build.add_argument("--seed", type=int, default=0)
    show = sub.add_parser("show", help="print a prompt list and its coverage")
    show.add_argument("path", type=Path)


def _print_report(report: dict) -> None:
    print(
        f"{report['prompts']} prompts, {report['phoneme_tokens']} phoneme tokens, "
        f"{report['phonemes_covered']}/{report['phonemes_total']} phonemes, {report['distinct_diphones']} distinct diphones"
    )
    print("rarest phonemes: " + ", ".join(f"{p} x{c}" for p, c in report["rarest_phonemes"]))
    if report["missing_phonemes"]:
        print("not covered: " + " ".join(report["missing_phonemes"]))


def run(args: argparse.Namespace) -> int:
    from ..collect.prompts import build_prompt_set, coverage_report, load_prompts, read_text_pool, save_prompts
    from ..phonology import Inventory, Lexicon

    inventory = Inventory.default()
    if args.action == "show":
        prompts = load_prompts(args.path)
        for prompt in prompts:
            print(f"{prompt.prompt_id}  {prompt.kind:8s} {prompt.text}")
        _print_report(coverage_report(prompts, inventory))
        return 0
    lexicon = Lexicon.core(inventory)
    if args.lexicon:
        lexicon = Lexicon.load(args.lexicon, inventory=inventory)
        lexicon.add_control_words(inventory)
    pool = read_text_pool(args.pool, args.pool_limit) if args.pool else None
    prompts = build_prompt_set(
        lexicon, inventory, n_words=args.words, n_sentences=args.sentences, pool=pool,
        max_words=args.max_words, seed=args.seed, clench_every=args.clench_every,
    )  # fmt: skip
    save_prompts(args.out, prompts)
    _print_report(coverage_report(prompts, inventory))
    print(f"-> {args.out}")
    return 0
