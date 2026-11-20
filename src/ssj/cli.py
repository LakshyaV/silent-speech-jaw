"""``ssj`` command-line entry point.

Sub-commands are discovered from :mod:`ssj.commands`: every module in that
package that defines ``NAME``, ``HELP``, ``configure(parser)`` and ``run(args)``
becomes a sub-command. Command modules keep heavy imports (``torch``,
``llama_cpp``) inside ``run`` so that ``ssj --help`` stays instant and the
runtime path never pays for the training stack.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import pkgutil
import sys
from collections.abc import Sequence
from types import ModuleType

from . import __version__


def _discover() -> list[ModuleType]:
    from . import commands

    found: list[ModuleType] = []
    for info in sorted(pkgutil.iter_modules(commands.__path__), key=lambda m: m.name):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{commands.__name__}.{info.name}")
        if all(hasattr(module, attr) for attr in ("NAME", "HELP", "configure", "run")):
            found.append(module)
    return sorted(found, key=lambda m: (getattr(m, "ORDER", 100), m.NAME))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ssj",
        description="Silent speech from a dual jaw/reference IMU: record, train, decode, run live.",
    )
    parser.add_argument("--version", action="version", version=f"ssj {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="more logging (-vv for debug)")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    for module in _discover():
        child = sub.add_parser(
            module.NAME,
            help=module.HELP,
            description=(module.__doc__ or module.HELP).strip(),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        module.configure(child)
        child.set_defaults(_run=module.run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    if not getattr(args, "_run", None):
        parser.print_help()
        return 0
    try:
        code = args._run(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    return int(code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
