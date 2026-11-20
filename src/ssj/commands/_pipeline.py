"""Shared options for commands that run the decoding pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .. import assets
from ._common import add_config_arguments, resolve_config, section


def add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bundle", type=Path, required=True, help="model bundle directory")
    parser.add_argument(
        "--lexicon",
        default=None,
        help="CMUdict-format lexicon (default: <assets>/lexicon/cmudict.dict, else bundled core)",
    )
    parser.add_argument(
        "--lm", default=None, help="n-gram model (default: <assets>/lm/en.npz, else a small built-in one)"
    )
    parser.add_argument("--llm", default=None, help="GGUF model (default: $SSJ_LLM or the only model in <assets>/llm)")
    parser.add_argument("--no-llm", action="store_true", help="decode with the n-gram model only")
    parser.add_argument("--gpu-layers", type=int, default=None, help="LLM layers to offload (-1 = all)")
    parser.add_argument("--assets", default=None, help="assets directory (default: ./assets or $SSJ_ASSETS)")
    add_config_arguments(parser)


def build_pipeline(args: argparse.Namespace):
    """Create a :class:`Pipeline` from command-line options and the ``pipeline`` config section."""
    from dataclasses import replace

    from ..config import from_dict
    from ..runtime.pipeline import Pipeline, PipelineConfig

    config = from_dict(PipelineConfig, section(resolve_config(args), "pipeline"))
    root = assets.assets_dir(args.assets)
    lexicon = (
        args.lexicon or config.lexicon or (str(assets.lexicon_path(root)) if assets.lexicon_path(root).exists() else "")
    )
    lm = args.lm or config.lm or (str(assets.lm_path(root)) if assets.lm_path(root).exists() else "")
    llm = config.llm
    if args.no_llm:
        llm = replace(llm, model="")
    else:
        found = args.llm or llm.model or (str(assets.find_llm(root)) if assets.find_llm(root) else "")
        llm = replace(llm, model=found)
    if args.gpu_layers is not None:
        llm = replace(llm, n_gpu_layers=args.gpu_layers)
    config = replace(config, lexicon=lexicon, lm=lm, llm=llm)
    return Pipeline.from_directory(args.bundle, config)
