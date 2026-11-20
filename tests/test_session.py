from __future__ import annotations

import socket
import threading

import numpy as np
import pytest

from ssj.io import (
    RAW_DTYPE,
    BridgeSource,
    DeviceInfo,
    ReplaySource,
    Session,
    SessionWriter,
    SourceError,
    TcpSource,
    Utterance,
    find_sessions,
    open_source,
)
from ssj.io.protocol import MODE_FIFO, encode_info, encode_sample


def make_raw(n: int = 400, period_us: int = 4800) -> np.ndarray:
    raw = np.zeros(2 * n, dtype=RAW_DTYPE)
    for sensor in (0, 1):
        part = raw[sensor::2]
        part["t_us"] = 1_000_000 + np.arange(n) * period_us + sensor * 700
        part["seq"] = np.arange(n)
        part["sensor"] = sensor
        part["flags"] = 3
        part["accel"][:, 2] = 8197 if sensor == 0 else -8197
        part["gyro"][:, 0] = np.arange(n) % 50
    return raw


def test_session_round_trip(tmp_path):
    raw = make_raw()
    info = DeviceInfo(firmware=(0, 1, 0), mode=MODE_FIFO)
    directory = tmp_path / "s01" / "a"
    with SessionWriter(directory, subject="s01", session="a", info=info, meta={"note": "left cheek"}) as writer:
        writer.append(raw[:300])
        writer.calibration("rest", 1_000_000, 1_500_000)
        writer.append(raw[300:])
        writer.utterance(Utterance("u0", "hello world", 1_600_000, 2_400_000, phonemes=["HH", "AH"]))
        writer.utterance(Utterance("u1", "bad take", 2_500_000, 2_900_000, status="bad"))
        with pytest.raises(ValueError):
            writer.calibration("dance", 0, 1)
    session = Session(directory)
    assert session.key == "s01/a" and session.meta["note"] == "left cheek" and session.meta["n_samples"] == len(raw)
    assert session.info == info
    assert np.array_equal(session.raw(), raw)
    assert [u.utt_id for u in session.utterances()] == ["u0"]
    assert [u.utt_id for u in session.utterances(include_bad=True)] == ["u0", "u1"]
    assert session.utterances()[0].phonemes == ["HH", "AH"]
    assert session.calibration_segments() == {"rest": (1_000_000, 1_500_000)}
    streams = session.streams()
    assert streams["jaw"][1][0, 2] == pytest.approx(1.0, abs=1e-3)
    assert streams["ref"][1][0, 2] == pytest.approx(-1.0, abs=1e-3)
    assert session.duration_s() == pytest.approx(399 * 4800e-6, rel=1e-2)
    assert [s.key for s in find_sessions(tmp_path)] == ["s01/a"]


def test_writer_refuses_to_overwrite_and_reader_tolerates_truncation(tmp_path):
    raw = make_raw(50)
    with SessionWriter(tmp_path / "x", subject="s", session="x", info=DeviceInfo()) as writer:
        writer.append(raw)
    with pytest.raises(FileExistsError):
        SessionWriter(tmp_path / "x", subject="s", session="x", info=DeviceInfo())
    samples = tmp_path / "x" / "samples.bin"
    samples.write_bytes(samples.read_bytes()[:-7])  # recording interrupted mid-frame
    assert len(Session(tmp_path / "x").raw()) == len(raw) - 1
    with pytest.raises(FileNotFoundError):
        Session(tmp_path / "missing")


def test_replay_source_delivers_everything_in_order():
    raw = make_raw(300)
    source = ReplaySource(raw, DeviceInfo(), chunk_s=0.05)
    chunks = []
    while not source.exhausted:
        chunks.append(source.read())
    out = np.concatenate(chunks)
    assert len(chunks) > 10
    assert np.array_equal(out, raw[np.argsort(raw["t_us"], kind="stable")])
    assert source.device_now_us() == int(raw["t_us"].max())
    assert len(source.read()) == 0


def test_bridge_source_decodes_base64_batches():
    import base64

    frames = [encode_info(DeviceInfo(mode=MODE_FIFO))] + [
        encode_sample(i % 2, i // 2, 5000 * i, (i, 0, 0), (0, i, 0)) for i in range(40)
    ]
    batches = [base64.b64encode(b"".join(frames[i : i + 7])).decode() for i in range(0, len(frames), 7)] + [
        "",
        "not base64!",
    ]
    calls: list[str] = []

    def call(method: str):
        calls.append(method)
        if method == "get_imu_status":
            return "sensors_ready=1"
        return [batches.pop(0)] if batches else ""  # some RPC versions wrap the value in a list

    source = BridgeSource(call, poll_interval=0.0)
    assert source.status_text() == "sensors_ready=1"
    assert source.wait_info().mode_name == "fifo"
    rows = [source.read() for _ in range(10)]
    assert sum(len(r) for r in rows) == 40
    assert source.stats.framing_errors == 1


def test_tcp_source_reads_relayed_frames():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    payload = encode_info(DeviceInfo()) + b"".join(
        encode_sample(0, i, 1000 * i, (0, 0, 0), (0, 0, 0)) for i in range(25)
    )

    def serve():
        client, _ = server.accept()
        client.sendall(payload[:100])
        client.sendall(payload[100:])
        client.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    with open_source(f"tcp:127.0.0.1:{port}") as source:
        assert isinstance(source, TcpSource)
        total = 0
        for _ in range(200):
            total += len(source.read())
            if source.exhausted:
                break
        assert total == 25 and source.info is not None
    thread.join(timeout=2)
    server.close()


def test_open_source_rejects_bad_specs(tmp_path):
    with pytest.raises(SourceError):
        open_source("carrier-pigeon")
    with pytest.raises(SourceError):
        open_source("replay:")
    with pytest.raises(SourceError):
        open_source("tcp:127.0.0.1:1")
