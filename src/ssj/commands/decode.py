"""Decode a stored session through the live code path.

    ssj decode --bundle runs/base/bundle --session data/raw/s01/20260301-1010
    ssj decode --bundle runs/base/bundle --session data/sim/sim/s07 --limit 20

The session's raw frames are replayed through exactly what ``ssj live`` runs:
streaming front-end, activity segmentation, recogniser, beam search and LLM
fusion. Nothing is taken from the recorded cue windows, so this shows how the
system behaves on the session as a continuous stream. When the session carries
prompts, each decoded segment is compared with the prompt it overlaps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ._pipeline import add_pipeline_arguments, build_pipeline

NAME = "decode"
HELP = "decode a recorded session as a continuous stream"
ORDER = 60


def configure(parser: argparse.ArgumentParser) -> None:
    add_pipeline_arguments(parser)
    parser.add_argument("--session", type=Path, required=True, help="session directory")
    parser.add_argument("--limit", type=int, default=None, help="stop after N decoded utterances")
    parser.add_argument("--hint", default="", help="situation description given to the LLM")
    parser.add_argument("--no-context", action="store_true", help="do not feed earlier sentences to the LLM")
    parser.add_argument(
        "--verbose-candidates", type=int, default=0, help="print the top N candidates of each utterance"
    )


def run(args: argparse.Namespace) -> int:
    from ..dsp.calibration import calibrate_session
    from ..eval.metrics import ErrorRate
    from ..io.session import Session
    from ..io.sources import ReplaySource
    from ..phonology import normalize_text
    from ..runtime.live import LiveDecoder, LiveEvent

    pipeline = build_pipeline(args)
    session = Session(args.session)
    calibration = calibrate_session(session, pipeline.bundle.frontend_config.fs_internal)
    print(pipeline.describe())
    print(f"session {session.key}: {session.duration_s():.0f} s, calibration {calibration.level}")
    utterances = session.utterances()
    skip_until = max((end for _, end in session.calibration_segments().values()), default=0) * 1e-6
    wer = ErrorRate()
    matched: set[str] = set()

    def reference_for(event: LiveEvent) -> list[str] | None:
        middle = (event.start_s + event.duration_s / 2) * 1e6
        for utterance in utterances:
            if utterance.t_start_us <= middle <= utterance.t_end_us and utterance.utt_id not in matched:
                matched.add(utterance.utt_id)
                return normalize_text(utterance.text)
        return None

    shown = 0

    def on_event(event: LiveEvent) -> None:
        nonlocal shown
        if event.start_s < skip_until:
            return  # movement during the calibration routine
        shown += 1
        result = event.result
        reference = reference_for(event)
        flag = "" if result.accepted else "  (uncertain)"
        print(
            f"\n[{event.start_s:7.1f}s +{event.duration_s:.1f}s] {result.text or '<nothing>'}{flag}   confidence {result.confidence:.2f}, {1000 * event.latency_s:.0f} ms"
        )
        print(f"    phonemes: {' '.join(result.phonemes)}")
        if event.control:
            print(f"    control:  {' '.join(event.control)}")
        if reference is not None:
            errors = wer.add(reference, list(result.words))
            print(f"    prompt:   {' '.join(reference)}   ({'match' if errors == 0 else f'{errors} word errors'})")
        for candidate in result.candidates[: args.verbose_candidates]:
            print(
                f"      {candidate.score:9.2f}  am {candidate.am:8.2f}  ngram {candidate.ngram:7.2f}  llm {candidate.llm:7.2f}  [{candidate.source}] {' '.join(candidate.words)}"
            )
        if args.no_context:
            decoder.history.clear()
        if args.limit and shown >= args.limit:
            raise KeyboardInterrupt

    source = ReplaySource(session)
    decoder = LiveDecoder(source, pipeline, calibration=calibration, on_event=on_event, hint=args.hint)
    decoder.run()
    if wer.sentences:
        missed = len(utterances) - len(matched) if not args.limit else 0
        print(
            f"\n{wer.sentences} prompted utterances decoded, WER {wer.rate:.3f}"
            + (f", {missed} prompts not detected" if missed else "")
        )
    pipeline.close()
    return 0
