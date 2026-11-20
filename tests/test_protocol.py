from __future__ import annotations

import numpy as np
import pytest

from ssj.io.protocol import (
    FLAGS_OK,
    MODE_FIFO,
    RAW_DTYPE,
    SENSOR_JAW,
    SENSOR_REF,
    DeviceInfo,
    DeviceStatus,
    FrameDecoder,
    cobs_decode,
    cobs_encode,
    crc16_ccitt,
    encode_frame,
    encode_info,
    encode_log,
    encode_sample,
    encode_status,
    split_sensors,
    to_physical,
)


def test_crc_reference_vector():
    assert crc16_ccitt(b"123456789") == 0x29B1  # CRC-16/CCITT-FALSE check value


@pytest.mark.parametrize("length", [0, 1, 2, 23, 253, 254, 255, 256, 600])
def test_cobs_round_trip(length):
    rng = np.random.default_rng(length)
    for zero_share in (0.0, 0.2, 1.0):
        data = bytes(0 if rng.random() < zero_share else int(rng.integers(1, 256)) for _ in range(length))
        encoded = cobs_encode(data)
        assert 0 not in encoded
        assert cobs_decode(encoded) == data


def test_sample_frame_round_trip():
    decoder = FrameDecoder()
    frame = encode_sample(SENSOR_REF, 7, 123456, (1, -2, 3), (-4, 5, -6), FLAGS_OK)
    assert frame.endswith(b"\x00") and frame.count(b"\x00") == 1
    rows = decoder.feed(frame)
    assert rows.dtype == RAW_DTYPE and len(rows) == 1
    row = rows[0]
    assert (row["sensor"], row["seq"], row["t_us"], row["flags"]) == (SENSOR_REF, 7, 123456, FLAGS_OK)
    assert row["accel"].tolist() == [1, -2, 3] and row["gyro"].tolist() == [-4, 5, -6]


def test_info_status_log_frames():
    decoder = FrameDecoder()
    info = DeviceInfo(
        firmware=(1, 2, 3), mode=MODE_FIFO, odr_hz=208.0, accel_g_per_lsb=0.000122, gyro_dps_per_lsb=0.0175
    )
    decoder.feed(encode_info(info) + encode_status(DeviceStatus(1000, 5, 1, 2, 3, 4)) + encode_log("jaw IMU not found"))
    assert decoder.info.firmware == (1, 2, 3) and decoder.info.mode_name == "fifo"
    assert decoder.info.accel_g_per_lsb == pytest.approx(0.000122)
    assert decoder.status == DeviceStatus(1000, 5, 1, 2, 3, 4)
    assert decoder.logs == ["jaw IMU not found"]
    assert DeviceInfo.from_dict(info.to_dict()) == info


def test_arbitrary_chunking_gives_the_same_samples():
    rng = np.random.default_rng(0)
    stream = b"".join(
        encode_sample(i % 2, i // 2, 1000 * i, rng.integers(-300, 300, 3), rng.integers(-300, 300, 3))
        for i in range(200)
    )
    whole = FrameDecoder().feed(stream)
    decoder = FrameDecoder()
    parts, cursor = [], 0
    while cursor < len(stream):
        step = int(rng.integers(1, 40))
        parts.append(decoder.feed(stream[cursor : cursor + step]))
        cursor += step
    pieces = np.concatenate(parts)
    assert len(whole) == 200 and np.array_equal(whole, pieces)


def test_corruption_is_detected_and_the_stream_recovers():
    def sample(seq: int) -> bytes:
        return encode_sample(SENSOR_JAW, seq, 10 * seq, (1, 2, 3), (4, 5, 6))

    corrupt = bytearray(sample(2))
    corrupt[5] ^= 0x40
    decoder = FrameDecoder()
    rows = decoder.feed(b"\x07\x09garbage\x00" + sample(1) + bytes(corrupt) + b"\x00\x00" + sample(3))
    assert rows["seq"].tolist() == [1, 3]
    assert decoder.stats.crc_errors + decoder.stats.framing_errors == 2
    assert decoder.stats.lost_samples[SENSOR_JAW] == 1
    # Noise without a delimiter swallows exactly the frame it runs into.
    rows = decoder.feed(b"\x05noise" + sample(4) + sample(5))
    assert rows["seq"].tolist() == [5]
    assert decoder.feed(encode_frame(0x7F, b"future frame type")).size == 0
    assert decoder.stats.unknown_frames == 1


def test_sequence_and_clock_unwrap():
    decoder = FrameDecoder()
    frames = b""
    for i in range(6):  # crosses both the u16 sequence and the u32 microsecond wrap
        frames += encode_sample(SENSOR_JAW, 65533 + i, 4294967000 + 100 * i, (0, 0, 0), (0, 0, 0))
    rows = decoder.feed(frames)
    assert np.all(np.diff(rows["seq"].astype(np.int64)) == 1)
    assert np.all(np.diff(rows["t_us"]) == 100)
    assert rows["t_us"][-1] > 2**32


def test_split_and_physical_units():
    decoder = FrameDecoder()
    rows = decoder.feed(
        encode_sample(SENSOR_REF, 0, 5, (0, 0, 8197), (0, 0, 0))
        + encode_sample(SENSOR_JAW, 0, 6, (8197, 0, 0), (1000, 0, 0))
    )
    parts = split_sensors(rows)
    assert len(parts[SENSOR_JAW]) == 1 and len(parts[SENSOR_REF]) == 1
    t, accel, gyro = to_physical(parts[SENSOR_JAW], DeviceInfo())
    assert t[0] == pytest.approx(6e-6)
    assert accel[0, 0] == pytest.approx(1.0, abs=1e-3)  # 8197 LSB at 0.122 mg/LSB is 1 g
    assert gyro[0, 0] == pytest.approx(17.5)
