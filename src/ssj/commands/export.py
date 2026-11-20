"""Export a checkpoint to a deployable ONNX bundle.

    ssj export --checkpoint runs/base/checkpoint.pt --out bundles/base
    ssj export --checkpoint runs/adapted/checkpoint.pt --out bundles/today --session s01/20260301-1010

The export is verified numerically against the checkpoint before the bundle is
written. ``--session`` bakes that session's input adapter into the graph.
"""

from __future__ import annotations

import argparse
from pathlib import Path

NAME = "export"
HELP = "export a checkpoint to an ONNX bundle"
ORDER = 41


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--session", default=None, help="bake this session's adapter into the model")
    parser.add_argument("--opset", type=int, default=17)


def run(args: argparse.Namespace) -> int:
    from ..models.export import export_bundle

    meta = export_bundle(args.checkpoint, args.out, session=args.session, opset=args.opset)
    model = meta["model"]
    print(
        f"{model['parameters']:,} parameters, {model['bytes'] / 1e6:.2f} MB, max deviation from checkpoint {model['max_export_error']:.1e}"
    )
    print(f"-> {args.out}")
    return 0
