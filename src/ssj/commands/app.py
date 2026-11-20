"""Assemble the Arduino UNO Q App Lab application.

    ssj app package --bundle runs/base/bundle --out dist/ssj-app
    ssj app package --bundle runs/base/bundle --lm assets/lm/en.npz \
                    --llm assets/llm/smollm2-360m-instruct-q4_k_m.gguf --out dist/ssj-app
    ssj app package --relay --out dist/ssj-relay        # board only relays frames to a laptop

The output folder is a complete App Lab app: the streamer sketch for the MCU,
the Linux-side program, a copy of the ``ssj`` runtime, and (unless ``--relay``)
the model bundle and language resources. Copy it to the board, open it in
Arduino App Lab and press Run.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

NAME = "app"
HELP = "package the Arduino UNO Q application"
ORDER = 70


def configure(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="action", required=True)
    package = sub.add_parser("package", help="build a deployable App Lab folder")
    package.add_argument("--out", type=Path, required=True)
    package.add_argument("--bundle", type=Path, default=None, help="model bundle to deploy (decode mode)")
    package.add_argument("--lexicon", type=Path, default=None)
    package.add_argument("--lm", type=Path, default=None)
    package.add_argument("--llm", type=Path, default=None, help="GGUF model to deploy")
    package.add_argument("--config", type=Path, default=None, help="tuned decoder weights (from `ssj tune`)")
    package.add_argument("--relay", action="store_true", help="relay-only app: no model, the laptop does the decoding")
    package.add_argument("--hint", default="", help="situation description given to the LLM on the board")
    package.add_argument("--port", type=int, default=5555, help="TCP port of the relay")
    package.add_argument("--repo", type=Path, default=None, help="repository root (default: located automatically)")


def _repo_root(explicit: Path | None) -> Path:
    root = explicit or Path(__file__).resolve().parents[3]
    if not (root / "firmware" / "ssj_streamer").is_dir() or not (root / "app").is_dir():
        raise SystemExit("could not find firmware/ and app/: run this from a source checkout or pass --repo")
    return root


def _board_lexicon(lexicon_path: Path, lm_path: Path | None):
    """The lexicon to ship: restricted to the language model's vocabulary when there is one.

    The decoder can only produce words known to both, and a full dictionary costs
    tens of megabytes of RAM that a small board should not spend on unreachable words.
    """
    from ..decode.ngram import NGramLM
    from ..phonology import Inventory, Lexicon

    keep = None
    if lm_path is not None and lm_path.suffix == ".npz":
        keep = set(NGramLM.load(lm_path).vocab)
    return Lexicon.load(lexicon_path, inventory=Inventory.default(), keep=keep)


def run(args: argparse.Namespace) -> int:
    root = _repo_root(args.repo)
    out: Path = args.out
    if not args.relay and args.bundle is None:
        raise SystemExit("--bundle is required unless --relay is given")
    if out.exists():
        shutil.rmtree(out)
    (out / "python").mkdir(parents=True)
    (out / "sketch").mkdir()

    shutil.copy(root / "app" / "app.yaml", out / "app.yaml")
    shutil.copy(root / "app" / "python" / "main.py", out / "python" / "main.py")
    shutil.copy(root / "app" / "python" / "requirements.txt", out / "python" / "requirements.txt")
    shutil.copy(root / "app" / "sketch" / "sketch.yaml", out / "sketch" / "sketch.yaml")
    firmware = root / "firmware" / "ssj_streamer"
    shutil.copy(firmware / "ssj_streamer.ino", out / "sketch" / "sketch.ino")
    for header in firmware.glob("*.h"):
        shutil.copy(header, out / "sketch" / header.name)

    shutil.copytree(root / "src" / "ssj", out / "python" / "ssj", ignore=shutil.ignore_patterns("__pycache__"))

    lines = [f"mode: {'relay' if args.relay else 'decode'}"]
    if not args.relay:
        shutil.copytree(args.bundle, out / "python" / "bundle")
        lines.append(f"bundle: {args.bundle}")
        assets = out / "python" / "assets"
        assets.mkdir()
        if args.lm:
            shutil.copy(args.lm, assets / "lm.npz")
            lines.append(f"n-gram: {args.lm}")
        if args.lexicon:
            lexicon = _board_lexicon(args.lexicon, args.lm)
            lexicon.save(assets / "lexicon.dict")
            lines.append(f"lexicon: {args.lexicon} ({len(lexicon):,} words)")
        if args.llm:
            shutil.copy(args.llm, assets / args.llm.name)
            lines.append(f"llm: {args.llm.name} ({args.llm.stat().st_size / 1e6:.0f} MB)")
        if args.config:
            shutil.copy(args.config, out / "python" / "config.yaml")
            lines.append(f"config: {args.config}")
    else:
        requirements = out / "python" / "requirements.txt"
        requirements.write_text("numpy>=1.26\n")  # the relay only shuttles bytes
    settings = [f"SSJ_MODE={'relay' if args.relay else 'decode'}", f"SSJ_RELAY_PORT={args.port}"]
    if args.hint:
        settings.append(f"SSJ_HINT={args.hint}")
    (out / "python" / "settings.env").write_text("\n".join(settings) + "\n")
    if args.port != 5555:  # the port has to be exposed by the app container as well
        manifest = out / "app.yaml"
        manifest.write_text(manifest.read_text().replace("  - 5555", f"  - {args.port}"))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print("\n".join(lines))
    print(f"-> {out}  ({size / 1e6:.1f} MB)")
    if args.relay:
        print(f"on the laptop: ssj record --source tcp:<board-ip>:{args.port} ...")
    return 0
