"""Binary wire protocol between the IMU streamer firmware and the host.

Framing
-------
Every frame is ``COBS(type | body | crc16)`` followed by a single ``0x00``
delimiter. COBS guarantees the delimiter never appears inside a frame, so the
host can resynchronise after any corruption by skipping to the next zero byte.
The CRC is CRC-16/CCITT-FALSE over ``type | body``, little-endian.

Frame types (all little-endian)
-------------------------------
``0x01 SAMPLE``  sensor:u8 seq:u16 t_us:u32 accel:3*i16 gyro:3*i16 flags:u8
``0x02 INFO``    proto:u8 fw:3*u8 mode:u8 sensors:u8 odr_hz:f32 accel_g_per_lsb:f32 gyro_dps_per_lsb:f32
``0x03 STATUS``  uptime_ms:u32 frames:u32 i2c_errors:u32 fifo_overruns:u32 tx_dropped:u32 loop_max_us:u32
``0x04 LOG``     utf-8 text

One SAMPLE frame carries one reading of one sensor. The two IMUs run on their
own oscillators, so they are deliberately *not* paired on the wire; the host
resamples each stream onto a common time grid.

Samples are transported as raw sensor LSBs. The scale factors travel in the
INFO frame, so changing a full-scale range in the firmware can never silently
change the units the host sees.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

PROTO_VERSION = 1

FRAME_SAMPLE = 0x01
FRAME_INFO = 0x02
FRAME_STATUS = 0x03
FRAME_LOG = 0x04

SENSOR_JAW = 0
SENSOR_REF = 1
SENSOR_NAMES = {SENSOR_JAW: "jaw", SENSOR_REF: "ref"}

FLAG_ACCEL_FRESH = 1 << 0
FLAG_GYRO_FRESH = 1 << 1
FLAG_READ_ERROR = 1 << 2
FLAG_FIFO_OVERRUN = 1 << 3
FLAG_LATE = 1 << 4
FLAGS_OK = FLAG_ACCEL_FRESH | FLAG_GYRO_FRESH

MODE_POLLED = 0
MODE_FIFO = 1
MODE_NAMES = {MODE_POLLED: "polled", MODE_FIFO: "fifo"}

#: Single-byte host -> device commands.
CMD_INFO = b"i"
CMD_START = b"s"
CMD_PAUSE = b"p"
CMD_RESET_COUNTERS = b"r"

_SAMPLE = struct.Struct("<BHI3h3hB")
_INFO = struct.Struct("<BBBBBBfff")
_STATUS = struct.Struct("<IIIIII")

#: On-disk and in-memory representation of a received sample. ``t_us`` and
#: ``seq`` are unwrapped by the receiver (the wire carries u32 / u16).
RAW_DTYPE = np.dtype(
    [
        ("t_us", "<i8"),
        ("seq", "<u4"),
        ("sensor", "u1"),
        ("flags", "u1"),
        ("accel", "<i2", (3,)),
        ("gyro", "<i2", (3,)),
    ]
)


# --------------------------------------------------------------------------- CRC
def _crc_table() -> tuple[int, ...]:
    table = []
    for byte in range(256):
        crc = byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
        table.append(crc & 0xFFFF)
    return tuple(table)


_CRC_TABLE = _crc_table()


def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection)."""
    for byte in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CRC_TABLE[((crc >> 8) ^ byte) & 0xFF]
    return crc


# -------------------------------------------------------------------------- COBS
def cobs_encode(data: bytes) -> bytes:
    out = bytearray()
    code_index = 0
    out.append(0)
    code = 1
    for byte in data:
        if byte == 0:
            out[code_index] = code
            code_index = len(out)
            out.append(0)
            code = 1
        else:
            out.append(byte)
            code += 1
            if code == 0xFF:
                out[code_index] = code
                code_index = len(out)
                out.append(0)
                code = 1
    out[code_index] = code
    return bytes(out)


def cobs_decode(data: bytes) -> bytes:
    out = bytearray()
    index = 0
    n = len(data)
    while index < n:
        code = data[index]
        if code == 0:
            raise ValueError("zero byte inside COBS frame")
        index += 1
        end = index + code - 1
        if end > n:
            raise ValueError("truncated COBS frame")
        out += data[index:end]
        index = end
        if code < 0xFF and index < n:
            out.append(0)
    return bytes(out)


# ------------------------------------------------------------------------ frames
@dataclass(frozen=True)
class DeviceInfo:
    """Static description of the streamer, sent in INFO frames."""

    proto: int = PROTO_VERSION
    firmware: tuple[int, int, int] = (0, 0, 0)
    mode: int = MODE_POLLED
    sensors: int = 0b11
    odr_hz: float = 208.0
    accel_g_per_lsb: float = 0.000122
    gyro_dps_per_lsb: float = 0.0175

    @property
    def mode_name(self) -> str:
        return MODE_NAMES.get(self.mode, f"mode{self.mode}")

    def to_dict(self) -> dict:
        return {
            "proto": self.proto,
            "firmware": ".".join(str(v) for v in self.firmware),
            "mode": self.mode_name,
            "sensors": self.sensors,
            "odr_hz": self.odr_hz,
            "accel_g_per_lsb": self.accel_g_per_lsb,
            "gyro_dps_per_lsb": self.gyro_dps_per_lsb,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DeviceInfo:
        modes = {name: value for value, name in MODE_NAMES.items()}
        firmware = tuple(int(v) for v in str(data.get("firmware", "0.0.0")).split("."))
        return cls(
            proto=int(data.get("proto", PROTO_VERSION)),
            firmware=(*firmware, 0, 0, 0)[:3],  # type: ignore[arg-type]
            mode=modes.get(str(data.get("mode", "polled")), MODE_POLLED),
            sensors=int(data.get("sensors", 0b11)),
            odr_hz=float(data.get("odr_hz", 208.0)),
            accel_g_per_lsb=float(data["accel_g_per_lsb"]),
            gyro_dps_per_lsb=float(data["gyro_dps_per_lsb"]),
        )


@dataclass(frozen=True)
class DeviceStatus:
    uptime_ms: int = 0
    frames: int = 0
    i2c_errors: int = 0
    fifo_overruns: int = 0
    tx_dropped: int = 0
    loop_max_us: int = 0


def encode_frame(frame_type: int, body: bytes) -> bytes:
    payload = bytes([frame_type]) + body
    return cobs_encode(payload + struct.pack("<H", crc16_ccitt(payload))) + b"\x00"


def encode_sample(sensor: int, seq: int, t_us: int, accel, gyro, flags: int = FLAGS_OK) -> bytes:
    body = _SAMPLE.pack(sensor, seq & 0xFFFF, t_us & 0xFFFFFFFF, *map(int, accel), *map(int, gyro), flags)
    return encode_frame(FRAME_SAMPLE, body)


def encode_info(info: DeviceInfo) -> bytes:
    body = _INFO.pack(
        info.proto, *info.firmware, info.mode, info.sensors, info.odr_hz, info.accel_g_per_lsb, info.gyro_dps_per_lsb
    )
    return encode_frame(FRAME_INFO, body)


def encode_status(status: DeviceStatus) -> bytes:
    body = _STATUS.pack(
        status.uptime_ms, status.frames, status.i2c_errors, status.fifo_overruns, status.tx_dropped, status.loop_max_us
    )
    return encode_frame(FRAME_STATUS, body)


def encode_log(text: str) -> bytes:
    return encode_frame(FRAME_LOG, text.encode("utf-8"))


class _Unwrap:
    """Extends a wrapping counter (u16 sequence, u32 microseconds) to 64 bits."""

    __slots__ = ("high", "last", "modulus")

    def __init__(self, bits: int):
        self.modulus = 1 << bits
        self.last: int | None = None
        self.high = 0

    def __call__(self, value: int) -> int:
        if self.last is not None:
            delta = value - self.last
            if delta < -(self.modulus // 2):  # counter wrapped
                self.high += self.modulus
            elif delta > self.modulus // 2 and self.high > 0:
                # A straggler from just before the wrap (the two sensors interleave).
                return self.high - self.modulus + value
        self.last = value
        return self.high + value


@dataclass
class DecoderStats:
    frames_ok: int = 0
    crc_errors: int = 0
    framing_errors: int = 0
    unknown_frames: int = 0
    lost_samples: dict[int, int] = field(default_factory=dict)


class FrameDecoder:
    """Incremental decoder: feed raw bytes, collect samples and metadata.

    The decoder owns the unwrap state of each sensor's sequence counter and
    clock, so the arrays it returns are directly usable as a continuous record.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._seq = {SENSOR_JAW: _Unwrap(16), SENSOR_REF: _Unwrap(16)}
        self._clock = _Unwrap(32)
        self._last_seq: dict[int, int] = {}
        self.info: DeviceInfo | None = None
        self.status: DeviceStatus | None = None
        self.logs: list[str] = []
        self.stats = DecoderStats()

    def feed(self, data: bytes) -> np.ndarray:
        """Consume bytes and return the newly completed samples as ``RAW_DTYPE``."""
        self._buffer += data
        rows: list[tuple] = []
        while True:
            end = self._buffer.find(0)
            if end < 0:
                break
            chunk = bytes(self._buffer[:end])
            del self._buffer[: end + 1]
            if not chunk:
                continue
            row = self._decode(chunk)
            if row is not None:
                rows.append(row)
        if len(self._buffer) > 4096:  # no delimiter in sight: line noise, drop it
            self.stats.framing_errors += 1
            self._buffer.clear()
        return np.array(rows, dtype=RAW_DTYPE) if rows else np.zeros(0, dtype=RAW_DTYPE)

    def _decode(self, chunk: bytes) -> tuple | None:
        try:
            payload = cobs_decode(chunk)
        except ValueError:
            self.stats.framing_errors += 1
            return None
        if len(payload) < 3:
            self.stats.framing_errors += 1
            return None
        body, (crc,) = payload[:-2], struct.unpack("<H", payload[-2:])
        if crc16_ccitt(body) != crc:
            self.stats.crc_errors += 1
            return None
        frame_type, body = body[0], body[1:]
        self.stats.frames_ok += 1

        if frame_type == FRAME_SAMPLE and len(body) == _SAMPLE.size:
            sensor, seq, t_us, ax, ay, az, gx, gy, gz, flags = _SAMPLE.unpack(body)
            if sensor not in self._seq:
                self.stats.unknown_frames += 1
                return None
            seq64 = self._seq[sensor](seq)
            previous = self._last_seq.get(sensor)
            if previous is not None and seq64 > previous + 1:
                self.stats.lost_samples[sensor] = self.stats.lost_samples.get(sensor, 0) + seq64 - previous - 1
            self._last_seq[sensor] = seq64
            return (self._clock(t_us), seq64, sensor, flags, (ax, ay, az), (gx, gy, gz))
        if frame_type == FRAME_INFO and len(body) == _INFO.size:
            proto, major, minor, patch, mode, sensors, odr, accel, gyro = _INFO.unpack(body)
            self.info = DeviceInfo(proto, (major, minor, patch), mode, sensors, odr, accel, gyro)
        elif frame_type == FRAME_STATUS and len(body) == _STATUS.size:
            self.status = DeviceStatus(*_STATUS.unpack(body))
        elif frame_type == FRAME_LOG:
            self.logs.append(body.decode("utf-8", errors="replace"))
        else:
            self.stats.unknown_frames += 1
        return None


def split_sensors(raw: np.ndarray) -> dict[int, np.ndarray]:
    """Split an interleaved record into per-sensor records (time-ordered)."""
    out = {}
    for sensor in (SENSOR_JAW, SENSOR_REF):
        part = raw[raw["sensor"] == sensor]
        out[sensor] = part[np.argsort(part["seq"], kind="stable")]
    return out


def to_physical(raw: np.ndarray, info: DeviceInfo) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert one sensor's records to ``(t_seconds, accel_g[N,3], gyro_dps[N,3])``."""
    t = raw["t_us"].astype(np.float64) * 1e-6
    accel = raw["accel"].astype(np.float64) * info.accel_g_per_lsb
    gyro = raw["gyro"].astype(np.float64) * info.gyro_dps_per_lsb
    return t, accel, gyro
