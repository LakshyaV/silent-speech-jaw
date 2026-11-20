"""Host-side checks of the firmware sources.

The firmware's framing code is plain C++ without Arduino dependencies, so it is
compiled natively here and its output is decoded by the Python implementation.
The two ends of the wire protocol are thereby tested against each other on
every run, without hardware.
"""

from __future__ import annotations

import base64
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ssj.io import protocol
from ssj.io.protocol import MODE_FIFO, SENSOR_JAW, SENSOR_REF, DeviceInfo, DeviceStatus, FrameDecoder, split_sensors

FIRMWARE = Path(__file__).resolve().parents[1] / "firmware" / "ssj_streamer"

HARNESS = r"""
#include <stdio.h>
#include "framing.h"
int main() {
  uint8_t body[ssj::MAX_BODY];
  uint8_t frame[ssj::MAX_FRAME];
  size_t n;
  for (int i = 0; i < 300; i++) {
    int16_t accel[3] = {(int16_t)(i * 7 - 900), (int16_t)0, (int16_t)(-i)};
    int16_t gyro[3] = {(int16_t)(i - 150), (int16_t)(256 * (i % 5)), (int16_t)0};
    n = ssj::packSample(body, i % 2, (uint16_t)(65500 + i / 2), 4294960000u + 4800u * (uint32_t)i, accel, gyro, 3);
    fwrite(frame, 1, ssj::buildFrame(ssj::FRAME_SAMPLE, body, n, frame), stdout);
  }
  n = ssj::packInfo(body, 0, 1, 0, ssj::MODE_FIFO, 3, 208.0f, 0.000122f, 0.0175f);
  fwrite(frame, 1, ssj::buildFrame(ssj::FRAME_INFO, body, n, frame), stdout);
  n = ssj::packStatus(body, 1, 2, 3, 4, 5, 6);
  fwrite(frame, 1, ssj::buildFrame(ssj::FRAME_STATUS, body, n, frame), stdout);
  const char* text = "hello";
  fwrite(frame, 1, ssj::buildFrame(ssj::FRAME_LOG, (const uint8_t*)text, 5, frame), stdout);
  char b64[64];
  const uint8_t raw[5] = {0, 1, 2, 250, 255};
  ssj::base64Encode(raw, 5, b64);
  fprintf(stderr, "%s", b64);
  return 0;
}
"""


def test_firmware_framing_matches_host_decoder(tmp_path):
    """Compile the firmware's framing code natively and decode its output in Python."""
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("no C++ compiler available")
    source = tmp_path / "harness.cpp"
    source.write_text(HARNESS)
    binary = tmp_path / "harness"
    subprocess.run([compiler, "-std=c++17", "-Wall", f"-I{FIRMWARE}", str(source), "-o", str(binary)], check=True)
    result = subprocess.run([str(binary)], check=True, capture_output=True)

    decoder = FrameDecoder()
    rows = decoder.feed(result.stdout)
    assert decoder.stats.crc_errors == 0 and decoder.stats.framing_errors == 0
    assert len(rows) == 300
    parts = split_sensors(rows)
    for sensor in (SENSOR_JAW, SENSOR_REF):
        assert np.all(np.diff(parts[sensor]["seq"].astype(np.int64)) == 1)  # across the u16 wrap
    assert np.all(np.diff(rows["t_us"]) == 4800)  # across the u32 wrap
    assert rows["accel"][10].tolist() == [10 * 7 - 900, 0, -10]
    assert rows["gyro"][10].tolist() == [10 - 150, 0, 0]
    assert decoder.info == DeviceInfo(1, (0, 1, 0), MODE_FIFO, 3, 208.0, pytest.approx(0.000122), pytest.approx(0.0175))
    assert decoder.status == DeviceStatus(1, 2, 3, 4, 5, 6)
    assert decoder.logs == ["hello"]
    assert base64.b64decode(result.stderr) == bytes([0, 1, 2, 250, 255])


def test_protocol_constants_agree():
    """Frame types, flags and scale factors are defined once per side; they must match."""
    header = (FIRMWARE / "framing.h").read_text()
    constants = {
        name: int(value, 0)
        for name, value in re.findall(r"constexpr uint8_t (\w+) = ([^;]+);", header)
        if "<<" not in value
    }
    constants.update(
        {name: 1 << int(shift) for name, shift in re.findall(r"constexpr uint8_t (\w+) = 1 << (\d+);", header)}
    )
    for name in ("PROTO_VERSION", "FRAME_SAMPLE", "FRAME_INFO", "FRAME_STATUS", "FRAME_LOG", "SENSOR_JAW", "SENSOR_REF",
                 "FLAG_ACCEL_FRESH", "FLAG_GYRO_FRESH", "FLAG_READ_ERROR", "FLAG_FIFO_OVERRUN", "FLAG_LATE", "MODE_POLLED", "MODE_FIFO"):  # fmt: skip
        assert constants[name] == getattr(protocol, name), name
    driver = (FIRMWARE / "lsm6dsox.h").read_text()
    accel = float(re.search(r"ACCEL_G_PER_LSB = ([0-9.]+)f", driver).group(1))
    gyro = float(re.search(r"GYRO_DPS_PER_LSB = ([0-9.]+)f", driver).group(1))
    assert accel == pytest.approx(DeviceInfo().accel_g_per_lsb) and gyro == pytest.approx(DeviceInfo().gyro_dps_per_lsb)
