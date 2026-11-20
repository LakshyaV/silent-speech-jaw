"""Render synthetic recording sessions with the articulatory simulator.

    ssj sim --out data/sim --sessions 12 --utterances 240
    ssj sim --out data/sim --pool assets/corpus/sentences.txt --sessions 20

Each session gets its own random sensor mounting, sensor clocks, biases, gains
and head-motion level, and is stored in exactly the format of a real recording,
so every later stage treats it the same way. Simulated sessions are used to
pre-train the recogniser before fine-tuning on recordings (``ssj train --init``)
and to exercise the whole pipeline without hardware.
"""

from __future__ import annotations

import argparse
from pathlib import Path

NAME = "sim"
HELP = "simulate recording sessions (pre-training data, hardware-free runs)"
ORDER = 25


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=Path, required=True, help="raw data root to write sessions into")
    parser.add_argument("--subject", default="sim")
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--utterances", type=int, default=200, help="utterances per session")
    parser.add_argument("--word-share", type=float, default=0.2, help="share of isolated-word utterances")
    parser.add_argument("--pool", type=Path, default=None, help="sentence pool (default: the built-in sentences)")
    parser.add_argument("--pool-limit", type=int, default=300000)
    parser.add_argument("--max-words", type=int, default=8)
    parser.add_argument("--lexicon", type=Path, default=None)
    parser.add_argument(
        "--clench-rate", type=float, default=0.05, help="share of utterances ending with the <clench> gesture"
    )
    parser.add_argument(
        "--mode", choices=["fifo", "polled"], default="fifo", help="firmware acquisition mode to emulate"
    )
    parser.add_argument(
        "--mount-spread", type=float, default=180.0, help="max mounting rotation in degrees (180 = arbitrary)"
    )
    parser.add_argument("--seed", type=int, default=0)


def run(args: argparse.Namespace) -> int:
    import numpy as np

    from ..collect.prompts import builtin_sentences, read_text_pool, word_drills
    from ..io.protocol import MODE_FIFO, MODE_POLLED
    from ..phonology import Inventory, Lexicon, normalize_text
    from ..sim import simulate_session

    inventory = Inventory.default()
    lexicon = Lexicon.core(inventory)
    if args.lexicon:
        lexicon = Lexicon.load(args.lexicon, inventory=inventory)
        lexicon.add_control_words(inventory)
    raw_pool = read_text_pool(args.pool, args.pool_limit) if args.pool else builtin_sentences()
    pool: list[str] = []
    seen: set[tuple[str, ...]] = set()
    for sentence in raw_pool:
        words = tuple(normalize_text(sentence))
        if 2 <= len(words) <= args.max_words and words not in seen and not lexicon.missing(words):
            seen.add(words)
            pool.append(" ".join(words))
    if not pool:
        raise SystemExit("no usable sentences in the pool (all contain out-of-lexicon words?)")
    drills = word_drills(lexicon, inventory, 400)
    rng = np.random.default_rng(args.seed)
    mode = MODE_FIFO if args.mode == "fifo" else MODE_POLLED
    n_words = round(args.utterances * args.word_share)
    print(f"sentence pool: {len(pool):,}; word drills: {len(drills)}")
    for index in range(args.sessions):
        texts = [drills[int(i)] for i in rng.integers(0, len(drills), n_words)]
        texts += [pool[int(i)] for i in rng.integers(0, len(pool), args.utterances - n_words)]
        rng.shuffle(texts)
        name = f"s{index:02d}"
        session = simulate_session(
            texts, lexicon, seed=args.seed * 1000 + index, mode=mode, mount_spread_deg=args.mount_spread,
            clench_rate=args.clench_rate, utt_prefix="u",
        )  # fmt: skip
        directory = session.write(args.out / args.subject / name, subject=args.subject, session=name)
        print(f"  {directory}  ({len(session.utterances)} utterances, {len(session.raw):,} frames)", flush=True)
    return 0
