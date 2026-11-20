"""Adapt a trained model to a new session from a short enrolment recording.

    ssj record --subject s01 --session today --words 20 --sentences 20
    ssj data build --raw data/raw --out data/processed
    ssj adapt --checkpoint runs/base/checkpoint.pt --data data/processed \
              --session s01/today --out runs/today

Only the session's affine input adapter is trained; the recogniser is frozen.
The adapted model is exported to ``<out>/bundle`` with the adapter baked in.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ._common import add_config_arguments, resolve_config, section

NAME = "adapt"
HELP = "fit a session adapter from enrolment data"
ORDER = 42


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--session", required=True, help="subject/session key of the enrolment session")
    parser.add_argument("--out", type=Path, required=True, help="run directory for the adapted model")
    add_config_arguments(parser)


def run(args: argparse.Namespace) -> int:
    from ..config import from_dict, save_json
    from ..models.export import export_bundle
    from ..train.adapt import AdaptConfig, adapt_session

    config = from_dict(AdaptConfig, section(resolve_config(args), "adapt"))
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "checkpoint.pt"
    summary = adapt_session(args.checkpoint, args.data, args.session, checkpoint, config)
    save_json(args.out / "adaptation.json", summary)
    export_bundle(checkpoint, args.out / "bundle", session=args.session)
    print(f"bundle -> {args.out / 'bundle'}")
    return 0
