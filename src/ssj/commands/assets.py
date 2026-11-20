"""Download language resources into the assets directory.

ssj assets fetch lexicon corpus llm        # everything, default LLM preset
ssj assets fetch llm --llm-preset tiny     # a smaller LLM for the board
ssj assets list
"""

from __future__ import annotations

import argparse

from .. import assets

NAME = "assets"
HELP = "fetch the pronunciation lexicon, a text corpus and a quantized LLM"
ORDER = 10


def configure(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="action", required=True)
    fetch = sub.add_parser("fetch", help="download resources")
    fetch.add_argument("what", nargs="+", choices=["lexicon", "corpus", "llm"])
    fetch.add_argument("--llm-preset", default="small", choices=sorted(assets.LLM_PRESETS))
    fetch.add_argument("--dir", default=None, help="assets directory (default: ./assets or $SSJ_ASSETS)")
    listing = sub.add_parser("list", help="show what is installed and which LLM presets exist")
    listing.add_argument("--dir", default=None)


def run(args: argparse.Namespace) -> int:
    root = assets.assets_dir(args.dir)
    if args.action == "list":
        llm = assets.find_llm(root)
        for label, path in (
            ("lexicon", assets.lexicon_path(root)),
            ("corpus", assets.corpus_path(root)),
            ("n-gram", assets.lm_path(root)),
        ):
            print(f"{label:8s} {'ok     ' if path.exists() else 'missing'} {path}")
        print(f"{'llm':8s} {'ok     ' if llm and llm.exists() else 'missing'} {llm or root / 'llm'}")
        print("\nLLM presets:")
        for name, preset in assets.LLM_PRESETS.items():
            print(f"  {name:6s} {preset.megabytes:5d} MB  {preset.filename}  - {preset.note}")
        return 0
    for item in args.what:
        print(f"fetching {item} ...")
        if item == "lexicon":
            path = assets.fetch_lexicon(root, assets.say)
        elif item == "corpus":
            path = assets.fetch_corpus(root, assets.say)
        else:
            path = assets.fetch_llm(root, args.llm_preset, assets.say)
        print(f"  -> {path}")
    return 0
