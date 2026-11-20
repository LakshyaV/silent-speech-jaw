// Wire framing shared by every transport: COBS + CRC-16/CCITT-FALSE.
//
// A frame on the wire is  COBS(type | body | crc_lo | crc_hi)  followed by 0x00.
// This header has no Arduino dependencies so that the host test-suite can
// compile it natively and check it against the Python implementation
// (src/ssj/io/protocol.py). Keep the two in sync; the protocol is specified in
// docs/protocol.md.
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <string.h>

namespace ssj {

constexpr uint8_t PROTO_VERSION = 1;

constexpr uint8_t FRAME_SAMPLE = 0x01;
constexpr uint8_t FRAME_INFO = 0x02;
constexpr uint8_t FRAME_STATUS = 0x03;
constexpr uint8_t FRAME_LOG = 0x04;

constexpr uint8_t SENSOR_JAW = 0;
constexpr uint8_t SENSOR_REF = 1;

constexpr uint8_t FLAG_ACCEL_FRESH = 1 << 0;
constexpr uint8_t FLAG_GYRO_FRESH = 1 << 1;
constexpr uint8_t FLAG_READ_ERROR = 1 << 2;
constexpr uint8_t FLAG_FIFO_OVERRUN = 1 << 3;
constexpr uint8_t FLAG_LATE = 1 << 4;

constexpr uint8_t MODE_POLLED = 0;
constexpr uint8_t MODE_FIFO = 1;

constexpr size_t MAX_BODY = 96;                 // largest frame body (LOG text)
constexpr size_t MAX_FRAME = MAX_BODY + 3 + 3;  // type + crc + COBS overhead + delimiter
constexpr size_t SAMPLE_BODY = 20;
constexpr size_t INFO_BODY = 18;
constexpr size_t STATUS_BODY = 24;

inline uint16_t crc16(const uint8_t* data, size_t n) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < n; i++) {
    crc ^= static_cast<uint16_t>(data[i]) << 8;
    for (uint8_t bit = 0; bit < 8; bit++) {
      crc = (crc & 0x8000) ? static_cast<uint16_t>((crc << 1) ^ 0x1021) : static_cast<uint16_t>(crc << 1);
    }
  }
  return crc;
}

// COBS-encode n bytes. `out` needs n + n / 254 + 1 bytes. Returns the encoded length
// (without delimiter).
inline size_t cobsEncode(const uint8_t* in, size_t n, uint8_t* out) {
  size_t write = 1;
  size_t codeIndex = 0;
  uint8_t code = 1;
  for (size_t read = 0; read < n; read++) {
    if (in[read] == 0) {
      out[codeIndex] = code;
      code = 1;
      codeIndex = write++;
    } else {
      out[write++] = in[read];
      if (++code == 0xFF) {
        out[codeIndex] = code;
        code = 1;
        codeIndex = write++;
      }
    }
  }
  out[codeIndex] = code;
  return write;
}

// Build a complete wire frame including the trailing delimiter. `out` needs
// MAX_FRAME bytes. Returns the number of bytes to transmit (0 if the body is too long).
inline size_t buildFrame(uint8_t type, const uint8_t* body, size_t n, uint8_t* out) {
  if (n > MAX_BODY) return 0;
  uint8_t raw[MAX_BODY + 3];
  raw[0] = type;
  memcpy(raw + 1, body, n);
  const uint16_t crc = crc16(raw, n + 1);
  raw[n + 1] = static_cast<uint8_t>(crc & 0xFF);
  raw[n + 2] = static_cast<uint8_t>(crc >> 8);
  size_t length = cobsEncode(raw, n + 3, out);
  out[length++] = 0;
  return length;
}

inline void put16(uint8_t* p, uint16_t v) {
  p[0] = static_cast<uint8_t>(v);
  p[1] = static_cast<uint8_t>(v >> 8);
}

inline void put32(uint8_t* p, uint32_t v) {
  p[0] = static_cast<uint8_t>(v);
  p[1] = static_cast<uint8_t>(v >> 8);
  p[2] = static_cast<uint8_t>(v >> 16);
  p[3] = static_cast<uint8_t>(v >> 24);
}

inline void putFloat(uint8_t* p, float v) {
  uint32_t bits;
  memcpy(&bits, &v, sizeof(bits));  // IEEE-754, transmitted little-endian
  put32(p, bits);
}

// SAMPLE body: sensor:u8 seq:u16 t_us:u32 accel:3*i16 gyro:3*i16 flags:u8
inline size_t packSample(uint8_t* body, uint8_t sensor, uint16_t seq, uint32_t tUs, const int16_t accel[3],
                         const int16_t gyro[3], uint8_t flags) {
  body[0] = sensor;
  put16(body + 1, seq);
  put32(body + 3, tUs);
  for (uint8_t i = 0; i < 3; i++) put16(body + 7 + 2 * i, static_cast<uint16_t>(accel[i]));
  for (uint8_t i = 0; i < 3; i++) put16(body + 13 + 2 * i, static_cast<uint16_t>(gyro[i]));
  body[19] = flags;
  return SAMPLE_BODY;
}

// INFO body: proto:u8 fw:3*u8 mode:u8 sensors:u8 odr_hz:f32 accel_g_per_lsb:f32 gyro_dps_per_lsb:f32
inline size_t packInfo(uint8_t* body, uint8_t major, uint8_t minor, uint8_t patch, uint8_t mode, uint8_t sensors,
                       float odrHz, float accelGPerLsb, float gyroDpsPerLsb) {
  body[0] = PROTO_VERSION;
  body[1] = major;
  body[2] = minor;
  body[3] = patch;
  body[4] = mode;
  body[5] = sensors;
  putFloat(body + 6, odrHz);
  putFloat(body + 10, accelGPerLsb);
  putFloat(body + 14, gyroDpsPerLsb);
  return INFO_BODY;
}

// STATUS body: uptime_ms frames i2c_errors fifo_overruns tx_dropped loop_max_us (all u32)
inline size_t packStatus(uint8_t* body, uint32_t uptimeMs, uint32_t frames, uint32_t i2cErrors, uint32_t fifoOverruns,
                         uint32_t txDropped, uint32_t loopMaxUs) {
  put32(body, uptimeMs);
  put32(body + 4, frames);
  put32(body + 8, i2cErrors);
  put32(body + 12, fifoOverruns);
  put32(body + 16, txDropped);
  put32(body + 20, loopMaxUs);
  return STATUS_BODY;
}

// Standard base64 with padding. `out` needs 4 * ((n + 2) / 3) + 1 bytes. Returns the
// length without the terminating NUL.
inline size_t base64Encode(const uint8_t* in, size_t n, char* out) {
  static const char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  size_t o = 0;
  for (size_t i = 0; i < n; i += 3) {
    const uint32_t b0 = in[i];
    const uint32_t b1 = (i + 1 < n) ? in[i + 1] : 0;
    const uint32_t b2 = (i + 2 < n) ? in[i + 2] : 0;
    const uint32_t triple = (b0 << 16) | (b1 << 8) | b2;
    out[o++] = alphabet[(triple >> 18) & 0x3F];
    out[o++] = alphabet[(triple >> 12) & 0x3F];
    out[o++] = (i + 1 < n) ? alphabet[(triple >> 6) & 0x3F] : '=';
    out[o++] = (i + 2 < n) ? alphabet[triple & 0x3F] : '=';
  }
  out[o] = '\0';
  return o;
}

}  // namespace ssj
