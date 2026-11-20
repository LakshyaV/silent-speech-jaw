from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from ssj import __version__
from ssj.cli import build_parser, main
from ssj.config import (
    ConfigError,
    apply_overrides,
    deep_merge,
    fingerprint,
    from_dict,
    load_yaml,
    save_yaml,
    to_dict,
)


def test_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_help_lists_commands(capsys):
    assert main([]) == 0
    assert "usage: ssj" in capsys.readouterr().out


def test_every_command_has_help():
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    for name, sub in (actions[0].choices if actions else {}).items():
        assert sub.format_help(), name


@dataclass(frozen=True)
class Inner:
    rate: float = 1.0
    taps: tuple[int, ...] = (1, 2)


@dataclass(frozen=True)
class Outer:
    name: str = "x"
    count: int = 3
    inner: Inner = field(default_factory=Inner)
    window: tuple[float, float] = (0.5, 1.5)
    maybe: int | None = None


def test_from_dict_round_trip():
    outer = from_dict(Outer, {"name": "y", "inner": {"rate": 2, "taps": [4, 8, 16]}, "window": [0, 1], "maybe": 5})
    assert outer == Outer("y", 3, Inner(2.0, (4, 8, 16)), (0.0, 1.0), 5)
    assert isinstance(outer.inner.rate, float)
    assert from_dict(Outer, to_dict(outer)) == outer


def test_from_dict_rejects_unknown_keys_and_wrong_types():
    with pytest.raises(ConfigError, match="unknown option"):
        from_dict(Outer, {"nmae": "typo"})
    with pytest.raises(ConfigError, match="inner"):
        from_dict(Outer, {"inner": {"rate": "fast"}})
    with pytest.raises(ConfigError):
        from_dict(Outer, {"count": True})
    with pytest.raises(ConfigError):
        from_dict(Outer, {"window": [1, 2, 3]})


def test_overrides_and_merge():
    base = {"train": {"epochs": 10, "lr": 0.1}, "model": {"streams": ["jaw"]}}
    out = apply_overrides(base, ["train.epochs=40", "model.streams=[rel, env]", "new.flag=true"])
    assert out["train"] == {"epochs": 40, "lr": 0.1}
    assert out["model"]["streams"] == ["rel", "env"]
    assert out["new"]["flag"] is True
    assert base["train"]["epochs"] == 10  # the input is not modified
    with pytest.raises(ConfigError):
        apply_overrides(base, ["no-equals-sign"])
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}


def test_fingerprint_is_stable_and_sensitive():
    assert fingerprint(Outer()) == fingerprint(Outer())
    assert fingerprint(Outer()) != fingerprint(Outer(count=4))
    assert fingerprint({"b": 1, "a": 2}) == fingerprint({"a": 2, "b": 1})


def test_yaml_round_trip(tmp_path):
    path = tmp_path / "nested" / "config.yaml"
    save_yaml(path, Outer(name="z"))
    assert from_dict(Outer, load_yaml(path)) == Outer(name="z")
