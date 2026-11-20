"""Tune decoder weights on the validation split.

    ssj tune --bundle runs/base/bundle --data data/processed --out tuned.yaml
    ssj eval --bundle runs/base/bundle --data data/processed -c tuned.yaml

First the first-pass parameters are searched (``tau``, n-gram weight, word
bonus), then the fusion weights of the CTC, n-gram and LLM scores. The result is
a config fragment to pass to ``eval``, ``decode`` and ``live`` with ``-c``.

The tuned ``tau`` tells you how much phonetic detail the sensor resolves beyond
the articulatory class: near 0 the phonemes within a class are indistinguishable
and the language models do all the disambiguation; near 1 the recogniser's
phoneme identities can be trusted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ._pipeline import add_pipeline_arguments, build_pipeline

NAME = "tune"
HELP = "tune decoding and fusion weights on held-out data"
ORDER = 51


def configure(parser: argparse.ArgumentParser) -> None:
    add_pipeline_arguments(parser)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--out", type=Path, required=True, help="YAML config fragment to write")
    parser.add_argument("--skip-first-pass", action="store_true")


def run(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from ..config import save_yaml
    from ..data.build import FeatureDataset
    from ..eval.report import evaluate
    from ..eval.tune import as_overrides, tune_first_pass, tune_fusion

    pipeline = build_pipeline(args)
    data = FeatureDataset(args.data)
    items = data.split(args.split)[: args.limit]
    if not items:
        raise SystemExit(f"the dataset has no {args.split!r} utterances")
    decoder = pipeline.decoder
    tau, beam = decoder.fusion.tau, decoder.beam
    if not args.skip_first_pass:
        print(f"first pass ({len(items)} utterances)")
        tau, beam, wer = tune_first_pass(pipeline, data, items, report=print)
        print(f"  best: tau={tau} lm_weight={beam.lm_weight} word_bonus={beam.word_bonus}  first-pass WER {wer:.3f}")
        decoder.beam = beam
        decoder.fusion = replace(decoder.fusion, tau=tau)
    print(f"fusion (LLM: {pipeline.backend.name})")
    _, records = evaluate(pipeline, data, items)
    fusion, before, after = tune_fusion(records, decoder.fusion)
    print(f"  WER {before:.3f} -> {after:.3f}  w_ngram={fusion.w_ngram} w_llm={fusion.w_llm} w_words={fusion.w_words}")
    save_yaml(args.out, {"pipeline": as_overrides(tau, beam, fusion)})
    print(f"-> {args.out}")
    pipeline.close()
    return 0
