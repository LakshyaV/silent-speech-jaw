"""Train the phoneme recogniser and export a deployable bundle.

    ssj train --data data/processed --out runs/base
    ssj train --data data/real --out runs/s01 --init runs/sim/checkpoint.pt   # fine-tune
    ssj train --data data/processed --out runs/control --control shuffled-labels

The best epoch (lowest validation phoneme error rate) is kept in
``<out>/checkpoint.pt`` and exported to ``<out>/bundle``. Options live in the
``model``, ``train`` and ``augment`` sections of the config:

    ssj train ... -s train.epochs=200 -s model.streams='[rel, env]'

Controls worth running on any new dataset: ``--control shuffled-labels`` must
not learn (otherwise labels leak through something other than articulation), and
``-s model.streams='[ref]'`` shows how much the reference sensor alone explains.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ._common import add_config_arguments, resolve_config, section

NAME = "train"
HELP = "train the phoneme recogniser"
ORDER = 40


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, required=True, help="dataset built by `ssj data build`")
    parser.add_argument("--out", type=Path, required=True, help="run directory")
    parser.add_argument("--init", type=Path, default=None, help="checkpoint to start from (pre-training, earlier run)")
    parser.add_argument("--control", choices=["shuffled-labels"], default=None, help="run a control experiment")
    parser.add_argument("--no-export", action="store_true", help="skip the ONNX bundle export")
    add_config_arguments(parser)


def run(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from ..config import from_dict
    from ..data.augment import AugmentConfig
    from ..models.encoder import ModelConfig
    from ..train.trainer import ExperimentConfig, TrainConfig, train

    config = resolve_config(args)
    experiment = ExperimentConfig(
        model=from_dict(ModelConfig, section(config, "model")),
        train=from_dict(TrainConfig, section(config, "train")),
        augment=from_dict(AugmentConfig, section(config, "augment")),
    )
    if args.control == "shuffled-labels":
        experiment = replace(experiment, train=replace(experiment.train, shuffle_labels=True))
    train(args.data, args.out, experiment, init=args.init)
    if not args.no_export:
        from ..models.export import export_bundle

        meta = export_bundle(args.out / "checkpoint.pt", args.out / "bundle")
        size = meta["model"]["bytes"] / 1e6
        print(f"bundle -> {args.out / 'bundle'} ({size:.1f} MB, export error {meta['model']['max_export_error']:.1e})")
    return 0
