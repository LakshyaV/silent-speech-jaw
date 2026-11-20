"""Sample sources: where raw IMU frames come from.

Every source yields the same thing - ``RAW_DTYPE`` arrays of both sensors in
arrival order plus a :class:`DeviceInfo` - so recording, offline decoding and
the live loop are written once against :class:`SampleSource`:

``SerialSource``   USB / UART link to the streamer firmware (laptop host)
``BridgeSource``   Arduino UNO Q RPC bridge (inference on the board's Linux side)
``TcpSource``      frames relayed over the network by the UNO Q app (laptop host)
``ReplaySource``   a stored or simulated session, optionally in real time
"""

from __future__ import annotations

import base64
import binascii
import socket
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

from .protocol import CMD_INFO, CMD_START, RAW_DTYPE, DecoderStats, DeviceInfo, DeviceStatus, FrameDecoder
from .session import Session


class SourceError(RuntimeError):
    pass


class SampleSource(ABC):
    """Common interface of all sample sources."""

    info: DeviceInfo | None = None

    def __init__(self) -> None:
        self._last_t_us: int | None = None
        self._last_host: float = 0.0
        self._pending: list[np.ndarray] = []

    @abstractmethod
    def _read(self) -> np.ndarray:
        """Return newly received samples from the transport (possibly empty)."""

    def read(self) -> np.ndarray:
        """Return newly available samples (possibly empty). May block briefly."""
        rows = self._read()
        if self._pending:
            rows = np.concatenate([*self._pending, rows])
            self._pending = []
        if len(rows):
            self._last_t_us = int(rows["t_us"].max())
            self._last_host = time.monotonic()
        return rows

    def close(self) -> None:  # noqa: B027 - optional hook
        pass

    @property
    def exhausted(self) -> bool:
        """True once a finite source has delivered everything."""
        return False

    def device_now_us(self) -> int:
        """Best estimate of the current device clock, for time-stamping host events."""
        if self._last_t_us is None:
            raise SourceError("no samples received yet")
        return self._last_t_us + int((time.monotonic() - self._last_host) * 1e6)

    def wait_info(self, timeout: float = 5.0) -> DeviceInfo:
        """Block until the device has described itself.

        Samples that arrive in the meantime are kept and returned by the next
        :meth:`read`, so nothing at the start of a stream is lost.
        """
        deadline = time.monotonic() + timeout
        while self.info is None:
            if time.monotonic() > deadline:
                raise SourceError("the device did not send an INFO frame; check firmware and wiring")
            rows = self._read()
            if len(rows):
                self._pending.append(rows)
        return self.info

    def __enter__(self) -> SampleSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def auto_detect_port() -> str:
    """Find the single likely Arduino serial port, or explain why not."""
    from serial.tools import list_ports

    ports = list(list_ports.comports())
    likely = sorted(
        {
            p.device
            for p in ports
            if any(tag in p.device.lower() for tag in ("usbmodem", "usbserial", "ttyacm", "ttyusb"))
            or "arduino" in f"{p.description} {p.manufacturer or ''}".lower()
        }
    )
    if len(likely) == 1:
        return likely[0]
    listing = "\n".join(f"  {p.device}  ({p.description})" for p in ports) or "  <none>"
    reason = "no likely Arduino port found" if not likely else "several candidate ports found"
    raise SourceError(f"{reason}; pass one explicitly.\nAvailable ports:\n{listing}")


class SerialSource(SampleSource):
    """Streamer firmware over a serial port (binary COBS frames)."""

    def __init__(self, port: str | None = None, baud: int = 921600, *, startup_wait: float = 1.5):
        super().__init__()
        try:
            import serial
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise SourceError("pyserial is required: pip install 'silent-speech-jaw[serial]'") from exc
        self.port = port or auto_detect_port()
        try:
            self._serial = serial.Serial(self.port, baudrate=baud, timeout=0.02, write_timeout=1.0)
        except serial.SerialException as exc:
            raise SourceError(f"could not open {self.port}: {exc}") from exc
        self._decoder = FrameDecoder()
        time.sleep(startup_wait)  # many boards reset when the port opens
        self._serial.reset_input_buffer()
        self.command(CMD_INFO)
        self.command(CMD_START)

    def command(self, byte: bytes) -> None:
        self._serial.write(byte)
        self._serial.flush()

    def _read(self) -> np.ndarray:
        waiting = self._serial.in_waiting
        data = self._serial.read(waiting if waiting else 1)
        rows = self._decoder.feed(data) if data else np.zeros(0, dtype=RAW_DTYPE)
        if self._decoder.info is not None:
            self.info = self._decoder.info
        return rows

    @property
    def stats(self) -> DecoderStats:
        return self._decoder.stats

    @property
    def status(self) -> DeviceStatus | None:
        return self._decoder.status

    def pop_logs(self) -> list[str]:
        logs, self._decoder.logs = self._decoder.logs, []
        return logs

    def close(self) -> None:
        self._serial.close()


class BridgeSource(SampleSource):
    """Streamer firmware behind the Arduino UNO Q RPC bridge.

    ``call`` is ``Bridge.call`` from ``arduino.app_utils``. The firmware exposes
    ``get_imu_batch`` which returns the pending protocol frames, base64-encoded
    (the RPC transports text). Decoding is therefore identical to the serial path.
    """

    def __init__(self, call: Callable[..., Any], *, poll_interval: float = 0.004):
        super().__init__()
        self._call = call
        self._decoder = FrameDecoder()
        self._poll_interval = poll_interval

    @staticmethod
    def _unwrap(payload: Any) -> str:
        while isinstance(payload, (list, tuple)) and len(payload) == 1:
            payload = payload[0]
        if payload is None:
            return ""
        if isinstance(payload, (bytes, bytearray)):
            return bytes(payload).decode("ascii", errors="ignore")
        return str(payload)

    def status_text(self) -> str:
        return self._unwrap(self._call("get_imu_status"))

    def _read(self) -> np.ndarray:
        text = self._unwrap(self._call("get_imu_batch")).strip()
        if not text:
            time.sleep(self._poll_interval)
            return np.zeros(0, dtype=RAW_DTYPE)
        try:
            data = base64.b64decode(text, validate=True)
        except (binascii.Error, ValueError):
            self._decoder.stats.framing_errors += 1
            return np.zeros(0, dtype=RAW_DTYPE)
        rows = self._decoder.feed(data)
        if self._decoder.info is not None:
            self.info = self._decoder.info
        return rows

    @property
    def stats(self) -> DecoderStats:
        return self._decoder.stats


class TcpSource(SampleSource):
    """Protocol frames over a TCP connection.

    The Arduino UNO Q has no byte-transparent serial link to a host computer, so
    for recording on a laptop the board's Linux side relays the frames it
    collects from the MCU over TCP (``app/`` in relay mode). The byte stream is
    the same framed protocol as on a serial port.
    """

    def __init__(self, host: str, port: int = 5555, *, timeout: float = 5.0):
        super().__init__()
        try:
            self._socket = socket.create_connection((host, port), timeout=timeout)
        except OSError as exc:
            raise SourceError(f"could not connect to {host}:{port}: {exc}") from exc
        self._socket.settimeout(0.05)
        self._decoder = FrameDecoder()
        self._closed = False

    @property
    def exhausted(self) -> bool:
        return self._closed

    def _read(self) -> np.ndarray:
        if self._closed:
            return np.zeros(0, dtype=RAW_DTYPE)
        try:
            data = self._socket.recv(65536)
        except TimeoutError:
            return np.zeros(0, dtype=RAW_DTYPE)
        except OSError:
            self._closed = True
            return np.zeros(0, dtype=RAW_DTYPE)
        if not data:
            self._closed = True
            return np.zeros(0, dtype=RAW_DTYPE)
        rows = self._decoder.feed(data)
        if self._decoder.info is not None:
            self.info = self._decoder.info
        return rows

    @property
    def stats(self) -> DecoderStats:
        return self._decoder.stats

    @property
    def status(self) -> DeviceStatus | None:
        return self._decoder.status

    def close(self) -> None:
        self._closed = True
        self._socket.close()


class ReplaySource(SampleSource):
    """Replay a recorded or simulated session through the live code path."""

    def __init__(
        self,
        raw: np.ndarray | Session,
        info: DeviceInfo | None = None,
        *,
        realtime: bool = False,
        chunk_s: float = 0.02,
    ):
        super().__init__()
        if isinstance(raw, Session):
            info = raw.info
            raw = raw.raw()
        if info is None:
            raise SourceError("ReplaySource needs a DeviceInfo")
        self.info = info
        order = np.argsort(raw["t_us"], kind="stable")
        self._raw = raw[order]
        self._cursor = 0
        self._chunk_us = int(chunk_s * 1e6)
        self._realtime = realtime
        self._wall_start: float | None = None

    @property
    def exhausted(self) -> bool:
        return self._cursor >= len(self._raw)

    def _read(self) -> np.ndarray:
        if self.exhausted:
            return np.zeros(0, dtype=RAW_DTYPE)
        t0 = int(self._raw["t_us"][0])
        if self._realtime:
            if self._wall_start is None:
                self._wall_start = time.monotonic()
            horizon = t0 + int((time.monotonic() - self._wall_start) * 1e6)
            if int(self._raw["t_us"][self._cursor]) > horizon:
                time.sleep(self._chunk_us * 1e-6)
                return np.zeros(0, dtype=RAW_DTYPE)
        else:
            horizon = int(self._raw["t_us"][self._cursor]) + self._chunk_us
        end = int(np.searchsorted(self._raw["t_us"], horizon, side="right"))
        end = max(end, self._cursor + 1)
        rows = self._raw[self._cursor : end]
        self._cursor = end
        return rows

    def device_now_us(self) -> int:
        if self._realtime:
            return super().device_now_us()
        if self._last_t_us is None:
            raise SourceError("no samples received yet")
        return self._last_t_us


def open_source(spec: str, **kwargs: Any) -> SampleSource:
    """Open a source from a short spec string.

    ``serial``, ``serial:/dev/cu.usbmodem1101``, ``tcp:192.168.1.50:5555`` or
    ``replay:path/to/session``.
    """
    kind, _, arg = spec.partition(":")
    if kind == "serial":
        return SerialSource(arg or None, **kwargs)
    if kind == "tcp":
        host, _, port = arg.rpartition(":")
        if not host:
            host, port = arg, "5555"
        if not host:
            raise SourceError("tcp source needs a host: tcp:<host>[:port]")
        return TcpSource(host, int(port))
    if kind == "replay":
        if not arg:
            raise SourceError("replay source needs a session directory: replay:<dir>")
        return ReplaySource(Session(arg), **kwargs)
    raise SourceError(f"unknown source {spec!r}; use serial[:port], tcp:<host>[:port] or replay:<dir>")
