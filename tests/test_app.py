from __future__ import annotations

import py_compile
from pathlib import Path

import pytest

from ssj.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_app_packaging(bundle_dir, tmp_path, capsys):
    out = tmp_path / "app"
    lm = tmp_path / "lm.npz"
    lm.write_bytes(b"placeholder")
    assert main(["app", "package", "--bundle", str(bundle_dir), "--lm", str(lm), "--out", str(out)]) == 0
    for path in ("app.yaml", "python/main.py", "python/requirements.txt", "python/ssj/runtime/live.py", "python/bundle/model.onnx",
                 "python/assets/lm.npz", "sketch/sketch.ino", "sketch/sketch.yaml", "sketch/framing.h", "sketch/lsm6dsox.h", "sketch/config.h"):  # fmt: skip
        assert (out / path).exists(), path
    assert not list(out.rglob("__pycache__"))
    assert (out / "python" / "settings.env").read_text().splitlines()[:2] == ["SSJ_MODE=decode", "SSJ_RELAY_PORT=5555"]
    assert "- 5555" in (out / "app.yaml").read_text()  # the relay port is exposed by the app container
    assert (out / "sketch" / "sketch.ino").read_text() == (
        ROOT / "firmware" / "ssj_streamer" / "ssj_streamer.ino"
    ).read_text()

    relay = tmp_path / "relay"
    assert main(["app", "package", "--relay", "--port", "6000", "--hint", "at home", "--out", str(relay)]) == 0
    assert (relay / "python" / "settings.env").read_text() == "SSJ_MODE=relay\nSSJ_RELAY_PORT=6000\nSSJ_HINT=at home\n"
    assert "- 6000" in (relay / "app.yaml").read_text()
    assert not (relay / "python" / "bundle").exists() and "relay" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["app", "package", "--out", str(tmp_path / "x")])
    py_compile.compile(str(ROOT / "app" / "python" / "main.py"), doraise=True)
