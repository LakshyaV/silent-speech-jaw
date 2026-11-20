"""Evaluate a bundle end to end on a built dataset.

    ssj eval --bundle runs/base/bundle --data data/processed
    ssj eval --bundle runs/base/bundle --data data/processed --split val --no-llm
    ssj eval --bundle runs/base/bundle --data data/processed -c tuned.yaml --out report.json

Reports the recogniser's phoneme and class error rates, the word error rate of
the first decoding pass, the oracle rate of the candidate pool and the final
word error rate after fusion, on a held-out split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._pipeline import add_pipeline_arguments, build_pipeline

NAME = "eval"
HELP = "measure PER and WER of a bundle on a held-out split"
ORDER = 50


def configure(parser: argparse.ArgumentParser) -> None:
    add_pipeline_arguments(parser)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N utterances")
    parser.add_argument("--samples", type=int, default=8, help="example outputs to print")
    parser.add_argument("--out", type=Path, default=None, help="write metrics as JSON")


def run(args: argparse.Namespace) -> int:
    from ..config import save_json
    from ..data.build import FeatureDataset
    from ..eval.report import evaluate, format_metrics, sample_lines

    pipeline = build_pipeline(args)
    data = FeatureDataset(args.data)
    items = data.split(args.split)[: args.limit]
    if not items:
        raise SystemExit(f"the dataset has no {args.split!r} utterances")
    print(pipeline.describe())

    def progress(done: int, total: int) -> None:
        if done % 20 == 0 or done == total:
            print(f"\r  decoding {done}/{total}", end="", file=sys.stderr, flush=True)

    metrics, records = evaluate(pipeline, data, items, progress=progress)
    print(file=sys.stderr)
    print(format_metrics(metrics, data.meta.get("split_notes", [])))
    for line in sample_lines(records, args.samples):
        print(line)
    if args.out:
        save_json(args.out, metrics)
    pipeline.close()
    return 0
