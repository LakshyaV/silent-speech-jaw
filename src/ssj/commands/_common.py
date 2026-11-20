"""Argument helpers shared by command modules."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..config import apply_overrides, deep_merge, load_yaml


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c", "--config", type=Path, action="append", default=[], help="YAML config file (repeatable; later files win)"
    )
    parser.add_argument(
        "-s",
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one option, e.g. -s train.epochs=40",
    )


def resolve_config(args: argparse.Namespace, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge defaults, ``--config`` files and ``--set`` overrides, in that order."""
    merged: dict[str, Any] = dict(defaults or {})
    for path in getattr(args, "config", []) or []:
        merged = deep_merge(merged, load_yaml(path))
    return apply_overrides(merged, getattr(args, "overrides", []) or [])


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name) or {}
    if not isinstance(value, dict):
        raise SystemExit(f"config section {name!r} must be a mapping")
    return value
