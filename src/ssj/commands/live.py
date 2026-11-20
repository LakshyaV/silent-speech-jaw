"""Run the recogniser live from the sensors.

    ssj live --bundle runs/base/bundle
    ssj live --bundle runs/today/bundle --source serial:/dev/cu.usbmodem1101 --hint "ordering coffee"

After a short calibration routine the system listens continuously: mouth a
sentence, pause, and the decoded text appears. Accepted sentences become context
for the LLM, so follow-up sentences are interpreted in the light of earlier ones;
``--hint`` gives it the situation up front.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._pipeline import add_pipeline_arguments, build_pipeline

NAME = "live"
HELP = "decode silent speech live from the sensors"
ORDER = 61


def configure(parser: argparse.ArgumentParser) -> None:
    add_pipeline_arguments(parser)
    parser.add_argument("--source", default="serial", help="serial[:port], tcp:<host>[:port] or replay:<session dir>")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument(
        "--no-calibration", action="store_true", help="skip the routine (uses --calibration or identity)"
    )
    parser.add_argument("--calibration", type=Path, default=None, help="reuse a calibration JSON from an earlier run")
    parser.add_argument("--save-calibration", type=Path, default=None)
    parser.add_argument("--hint", default="", help="situation description given to the LLM")
    parser.add_argument("--log", type=Path, default=None, help="append decoded utterances as JSON lines")
    parser.add_argument("--seconds", type=float, default=None, help="stop after this long")


def run(args: argparse.Namespace) -> int:
    from ..config import load_json, save_json
    from ..dsp.calibration import Calibration
    from ..io.sources import open_source
    from ..runtime.live import LiveDecoder, LiveEvent, run_calibration

    pipeline = build_pipeline(args)
    print(pipeline.describe())
    kwargs = {"baud": args.baud} if args.source.startswith("serial") else {}
    log = args.log.open("a", encoding="utf-8") if args.log else None

    def on_event(event: LiveEvent) -> None:
        result = event.result
        if not result.words:
            return
        marker = ">>>" if result.accepted else " ? "
        print(f"{marker} {result.text}    [{result.confidence:.2f}, {1000 * event.latency_s:.0f} ms]", flush=True)
        if event.control:
            print(f"    control: {' '.join(event.control)}", flush=True)
        if log:
            log.write(
                json.dumps(
                    {
                        "t": round(event.start_s, 2),
                        "text": result.text,
                        "words": list(result.words),
                        "confidence": round(result.confidence, 3),
                        "accepted": result.accepted,
                        "phonemes": result.phonemes,
                    }
                )
                + "\n"
            )
            log.flush()

    with open_source(args.source, **kwargs) as source:
        info = source.wait_info()
        print(f"device: firmware {'.'.join(map(str, info.firmware))}, {info.mode_name} mode, {info.odr_hz:.0f} Hz")
        if args.calibration:
            calibration = Calibration.from_dict(load_json(args.calibration))
        elif args.no_calibration:
            calibration = Calibration.identity()
        else:
            calibration = run_calibration(source, pipeline.bundle.frontend_config.fs_internal)
            if calibration.level != "full":
                print("warning: calibration is incomplete - accuracy will suffer; repeat with larger, slower movements")
        if args.save_calibration:
            save_json(args.save_calibration, calibration.to_dict())
        print("\nlistening - mouth a sentence, then pause.  Ctrl+C to stop.\n")
        LiveDecoder(source, pipeline, calibration=calibration, on_event=on_event, hint=args.hint).run(args.seconds)
    if log:
        log.close()
    pipeline.close()
    return 0
