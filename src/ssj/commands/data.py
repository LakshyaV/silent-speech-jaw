"""Build a feature dataset from recorded (or simulated) sessions.

    ssj data build --raw data/raw --out data/processed
    ssj data build --raw data/raw --out data/processed -s dataset.test_sessions='[s01/20260301-1010]'
    ssj data info data/processed

Every session is calibrated from its own routine, run through the streaming
front-end as one continuous signal and cut into utterances. Splits are made by
session; ``data info`` shows them and says so when a split had to be made inside
a single session.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ._common import add_config_arguments, resolve_config, section

NAME = "data"
HELP = "turn sessions into a training-ready feature dataset"
ORDER = 30


def configure(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build", help="process sessions into features")
    build.add_argument("--raw", type=Path, required=True, help="root containing <subject>/<session>/ directories")
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--lexicon", type=Path, default=None)
    build.add_argument("--inventory", type=Path, default=None, help="inventory YAML (e.g. from `ssj inventory refine`)")
    add_config_arguments(build)
    info = sub.add_parser("info", help="summarise a built dataset")
    info.add_argument("path", type=Path)


def _summary(meta: dict) -> None:
    splits = meta["splits"]
    hours = meta["frames"] / meta["frontend"]["fs"] / 3600
    print(
        f"utterances: train {splits['train']}, val {splits['val']}, test {splits['test']}  ({hours * 60:.1f} min of articulation)"
    )
    print(f"front-end fingerprint: {meta['frontend_fingerprint']}")
    for key, session in meta["sessions"].items():
        level = session["calibration"]["quality"].get("level", "none")
        print(f"  {key:28s} {session['split']:9s} {session['utterances']:4d} utterances   calibration: {level}")
    if meta.get("dropped"):
        print("dropped: " + ", ".join(f"{k} {v}" for k, v in meta["dropped"].items()))
    for note in meta.get("split_notes", []):
        print(f"note: {note}")


def run(args: argparse.Namespace) -> int:
    from ..config import from_dict, load_json
    from ..data.build import DatasetConfig, build_dataset
    from ..phonology import Inventory, Lexicon

    if args.action == "info":
        _summary(load_json(args.path / "meta.json"))
        return 0
    logging.getLogger("ssj").setLevel(logging.INFO)
    config = from_dict(DatasetConfig, section(resolve_config(args), "dataset"))
    inventory = Inventory.from_yaml(args.inventory) if args.inventory else Inventory.default()
    lexicon = Lexicon.core(inventory)
    if args.lexicon:
        lexicon = Lexicon.load(args.lexicon, inventory=inventory)
        lexicon.add_control_words(inventory)
    meta = build_dataset(args.raw, args.out, lexicon, inventory, config)
    _summary(meta)
    print(f"-> {args.out}")
    return 0
