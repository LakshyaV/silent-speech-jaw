#!/usr/bin/env python3
"""silent-speech-jaw on the Arduino UNO Q.

STM32 side:   samples both IMUs and queues framed samples (firmware/ssj_streamer).
Linux side:   this program. It collects the frames over the RPC bridge and either

  decode  runs the whole recogniser on the board: calibration, streaming
          front-end, phoneme model (ONNX), beam search and the quantized LLM;
  relay   forwards the raw frames over TCP so that a laptop can record
          sessions or decode with a larger model (``--source tcp:<board>:5555``).

Settings are read from ``settings.env`` next to this file (written by
``ssj app package``); environment variables of the same name take precedence:

  SSJ_MODE        decode | relay                     (default: decode)
  SSJ_BUNDLE      model bundle directory             (default: ./bundle)
  SSJ_LEXICON     pronunciation lexicon              (default: ./assets/lexicon.dict if present)
  SSJ_LM          n-gram model                       (default: ./assets/lm.npz if present)
  SSJ_LLM         GGUF model                         (default: the .gguf in ./assets, if any)
  SSJ_CONFIG      YAML with tuned decoder weights    (default: ./config.yaml if present)
  SSJ_HINT        situation description for the LLM
  SSJ_RELAY_PORT  TCP port in relay mode             (default: 5555)

This folder is assembled by ``ssj app package``; do not edit the copies of the
``ssj`` package or the sketch here, edit them in the repository.
"""

from __future__ import annotations

import base64
import binascii
import os
import socket
import threading
import time
import traceback
from pathlib import Path

from arduino.app_utils import App, Bridge

HERE = Path(__file__).resolve().parent


def load_settings(path: Path) -> None:
    """Apply ``KEY=VALUE`` lines as defaults for the environment."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() and not key.lstrip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip())


load_settings(HERE / "settings.env")
MODE = os.getenv("SSJ_MODE", "decode").strip().lower()


def log(message: str = "") -> None:
    print(message, flush=True)


def wait_for_bridge() -> None:
    """Block until the MCU answers; it may still be booting when Linux starts."""
    while True:
        try:
            status = Bridge.call("get_imu_status")
            while isinstance(status, (list, tuple)) and len(status) == 1:
                status = status[0]
            log(f"MCU ready: {status}")
            return
        except Exception as exc:
            log(f"waiting for the MCU bridge: {exc}")
            time.sleep(1.0)


def first_existing(*candidates: Path) -> str:
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return ""


def run_relay() -> None:
    """Serve the raw frame stream to one TCP client at a time."""
    port = int(os.getenv("SSJ_RELAY_PORT", "5555"))
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", port))
    server.listen(1)
    log(f"relay: listening on port {port} - connect with `--source tcp:<this board>:{port}`")
    while True:
        client, address = server.accept()
        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        log(f"relay: client {address[0]} connected")
        Bridge.call("get_imu_status")  # makes the MCU queue a fresh INFO frame
        sent = 0
        try:
            while True:
                payload = Bridge.call("get_imu_batch")
                while isinstance(payload, (list, tuple)) and len(payload) == 1:
                    payload = payload[0]
                text = (
                    payload.decode("ascii", "ignore") if isinstance(payload, (bytes, bytearray)) else str(payload or "")
                )
                if not text:
                    time.sleep(0.004)
                    continue
                try:
                    data = base64.b64decode(text, validate=True)
                except (binascii.Error, ValueError):
                    continue
                client.sendall(data)
                sent += len(data)
        except OSError:
            log(f"relay: client disconnected after {sent / 1e6:.1f} MB")
        finally:
            client.close()


def run_decode() -> None:
    from dataclasses import replace

    from ssj.config import from_dict, load_yaml
    from ssj.io.sources import BridgeSource
    from ssj.runtime.live import LiveDecoder, LiveEvent, run_calibration
    from ssj.runtime.pipeline import Pipeline, PipelineConfig

    config_file = os.getenv("SSJ_CONFIG") or first_existing(HERE / "config.yaml")
    section = (load_yaml(config_file).get("pipeline") or {}) if config_file else {}
    config = from_dict(PipelineConfig, section)
    assets = HERE / "assets"
    models = sorted(assets.glob("*.gguf")) if assets.is_dir() else []
    config = replace(
        config,
        lexicon=os.getenv("SSJ_LEXICON") or config.lexicon or first_existing(assets / "lexicon.dict"),
        lm=os.getenv("SSJ_LM") or config.lm or first_existing(assets / "lm.npz"),
        llm=replace(config.llm, model=os.getenv("SSJ_LLM") or config.llm.model or (str(models[0]) if models else "")),
    )
    pipeline = Pipeline.from_directory(os.getenv("SSJ_BUNDLE") or HERE / "bundle", config)
    log(pipeline.describe())

    source = BridgeSource(Bridge.call)
    info = source.wait_info(timeout=30.0)
    log(f"stream: {info.mode_name} mode, {info.odr_hz:.0f} Hz")
    calibration = run_calibration(source, pipeline.bundle.frontend_config.fs_internal, say=log)

    def on_event(event: LiveEvent) -> None:
        result = event.result
        if not result.words:
            return
        marker = ">>>" if result.accepted else " ? "
        log(f"{marker} {result.text}    [{result.confidence:.2f}, {1000 * event.latency_s:.0f} ms]")
        handle_sentence(result.text, result.accepted, event.control)

    log("\nlistening - mouth a sentence, then pause.\n")
    LiveDecoder(source, pipeline, calibration=calibration, on_event=on_event, hint=os.getenv("SSJ_HINT", "")).run()


def handle_sentence(text: str, accepted: bool, control: tuple[str, ...]) -> None:
    """Routing hook: act on decoded sentences here (speech synthesis, UI, actuators)."""


def worker() -> None:
    try:
        wait_for_bridge()
        if MODE == "relay":
            run_relay()
        else:
            run_decode()
    except Exception:
        log("\nsilent-speech-jaw stopped with an error:")
        traceback.print_exc()
        while True:
            time.sleep(60)


log(f"starting silent-speech-jaw ({MODE} mode)")
threading.Thread(target=worker, name="ssj-worker", daemon=True).start()
App.run()
