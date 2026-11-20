"""Inspect the phoneme inventory and refine its classes from measured confusions.

    ssj inventory show
    ssj inventory confusions --bundle runs/base/bundle --data data/processed
    ssj inventory refine --bundle runs/base/bundle --data data/processed --out configs/classes.yaml
    ssj data build --raw data/raw --out data/refined --inventory configs/classes.yaml

The built-in articulatory classes encode what a jaw sensor is expected to
separate. ``refine`` replaces that expectation with a measurement: phonemes the
trained recogniser cannot tell apart are merged into one class, and the rest are
split out. Rebuilding the dataset and retraining with the refined classes
focuses the auxiliary loss and the decoder on distinctions that actually exist.
"""

from __future__ import annotations

import argparse
from pathlib import Path

NAME = "inventory"
HELP = "show the phoneme classes or refine them from confusions"
ORDER = 52


def configure(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="action", required=True)
    show = sub.add_parser("show", help="print the inventory")
    show.add_argument("--inventory", type=Path, default=None)
    for name, text in (
        ("confusions", "print the most frequent phoneme confusions"),
        ("refine", "derive classes from confusions"),
    ):
        child = sub.add_parser(name, help=text)
        child.add_argument("--bundle", type=Path, required=True)
        child.add_argument("--data", type=Path, required=True)
        child.add_argument("--split", default="val", choices=["train", "val", "test"])
        child.add_argument("--limit", type=int, default=None)
        if name == "refine":
            child.add_argument(
                "--threshold", type=float, default=0.25, help="minimum confusability to merge two classes"
            )
            child.add_argument("--min-support", type=int, default=20)
            child.add_argument("--out", type=Path, required=True)
        else:
            child.add_argument("--top", type=int, default=20)


def run(args: argparse.Namespace) -> int:
    from ..phonology import Inventory

    if args.action == "show":
        inventory = Inventory.from_yaml(args.inventory) if args.inventory else Inventory.default()
        print(
            f"{inventory.size - 1} phonemes in {inventory.n_classes - 1} classes (fingerprint {inventory.fingerprint})"
        )
        for name in inventory.classes:
            print(f"  {name:16s} {' '.join(inventory.members(name))}")
        return 0

    from ..config import save_yaml
    from ..data.build import FeatureDataset
    from ..eval.confusion import confusion, describe_classes, refine_classes
    from ..eval.report import phoneme_sequences
    from ..runtime.bundle import Bundle
    from ..runtime.pipeline import Pipeline, PipelineConfig

    bundle = Bundle.load(args.bundle)
    pipeline = Pipeline(bundle, PipelineConfig())
    data = FeatureDataset(args.data)
    items = data.split(args.split)[: args.limit]
    if not items:
        raise SystemExit(f"the dataset has no {args.split!r} utterances")
    references, recognised = phoneme_sequences(pipeline, data, items)
    conf = confusion(references, recognised, bundle.inventory.phonemes)
    if args.action == "confusions":
        print("reference -> recognised   count   share of reference")
        for ref, hyp, count, rate in conf.top_confusions(args.top):
            print(f"  {ref:>6s} -> {hyp:<6s} {count:7d}   {100 * rate:5.1f}%")
        recall = conf.recall()
        worst = sorted(zip(conf.symbols, recall, conf.support), key=lambda x: x[1])[:8]
        print("lowest recall: " + ", ".join(f"{s} {100 * r:.0f}% (n={int(n)})" for s, r, n in worst if n > 0))
        return 0
    mapping = refine_classes(conf, bundle.inventory, threshold=args.threshold, min_support=args.min_support)
    refined = bundle.inventory.with_class_map(mapping)
    save_yaml(args.out, refined.to_dict())
    print(f"{bundle.inventory.n_classes - 1} classes -> {refined.n_classes - 1} classes")
    print(describe_classes(mapping))
    print(f"-> {args.out}")
    return 0
