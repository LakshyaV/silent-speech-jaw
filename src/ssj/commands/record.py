"""Record a labelled session with guided prompts.

    ssj record --subject s01 --prompts prompts/session01.jsonl
    ssj record --subject s01 --source serial:/dev/cu.usbmodem1101 --auto

The session starts with a link check and the three-phase calibration routine,
then cues each prompt: keep still, GO, relax. Every trial is checked on the
spot and can be redone. Sessions are written to ``<out>/<subject>/<session>/``
and can be interrupted at any time without losing what was recorded.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from ._common import add_config_arguments, resolve_config, section

NAME = "record"
HELP = "record a calibrated, prompted session"
ORDER = 21


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--subject", required=True, help="wearer id, e.g. s01")
    parser.add_argument("--session", default=None, help="session name (default: a timestamp)")
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--source", default="serial", help="serial[:port], tcp:<host>[:port] or replay:<session dir>")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument(
        "--prompts", type=Path, default=None, help="prompt list from `ssj prompts build` (default: built on the fly)"
    )
    parser.add_argument("--words", type=int, default=40)
    parser.add_argument("--sentences", type=int, default=80)
    parser.add_argument(
        "--seed", type=int, default=None, help="prompt selection seed (default: derived from the session name)"
    )
    parser.add_argument(
        "--auto", action="store_true", help="hands-free: accept clean trials and retry flagged ones automatically"
    )
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument(
        "--note", default="", help="free-text note stored with the session (mounting, tape, posture ...)"
    )
    add_config_arguments(parser)


def run(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from ..collect.prompts import build_prompt_set, load_prompts
    from ..collect.qc import QCConfig
    from ..collect.recorder import Recorder, RecorderConfig
    from ..config import from_dict
    from ..dsp.activity import SegmenterConfig
    from ..dsp.frontend import FrontendConfig
    from ..io.session import SessionWriter
    from ..io.sources import open_source
    from ..phonology import Inventory, Lexicon

    config = resolve_config(args)
    dataset = section(config, "dataset")
    recorder_config = from_dict(RecorderConfig, section(config, "recorder"))
    if args.auto:
        recorder_config = replace(recorder_config, auto=True)
    session = args.session or datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = args.out / args.subject / session

    if args.prompts:
        prompts = load_prompts(args.prompts)
    else:
        inventory = Inventory.default()
        seed = args.seed if args.seed is not None else sum(session.encode()) % 100000
        prompts = build_prompt_set(
            Lexicon.core(inventory), inventory, n_words=args.words, n_sentences=args.sentences, seed=seed
        )

    kwargs = {"baud": args.baud} if args.source.startswith("serial") else {}
    with open_source(args.source, **kwargs) as source:
        info = source.wait_info()
        print(f"device: firmware {'.'.join(map(str, info.firmware))}, {info.mode_name} mode, {info.odr_hz:.0f} Hz")
        with SessionWriter(
            directory,
            subject=args.subject,
            session=session,
            info=info,
            meta={"note": args.note, "prompts": len(prompts)},
        ) as writer:
            recorder = Recorder(
                source,
                writer,
                config=recorder_config,
                frontend_config=from_dict(FrontendConfig, dataset.get("frontend")),
                qc_config=from_dict(QCConfig, section(config, "qc")),
                segmenter=from_dict(SegmenterConfig, dataset.get("segmenter")),
            )
            counts = recorder.run(prompts, calibrate=not args.no_calibration)
            extra = {"counts": counts}
            if recorder.calibration is not None:
                extra["calibration"] = recorder.calibration.to_dict()
            writer.close(**extra)
    print(f"session saved to {directory}")
    return 0
