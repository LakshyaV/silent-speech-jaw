// silent-speech-jaw: dual-IMU streamer.
//
// Streams raw accelerometer and gyroscope samples of two LSM6DSOX IMUs - one on
// the jaw, one behind the ear as a head-motion reference - to a host that does
// everything else. The firmware is deliberately free of vocabulary, labels and
// trial logic: it measures and time-stamps, nothing more.
//
// Each sample is sent as one binary frame per sensor (see framing.h and
// docs/protocol.md). The sensors run on their own clocks, so they are never
// paired here; the host resamples both onto a common time base.
//
// Host commands (single bytes, serial transport):
//   i  send an INFO frame      s  start streaming      p  pause      r  reset counters

#include <Wire.h>

#include "config.h"
#include "framing.h"
#include "lsm6dsox.h"

#if SSJ_TRANSPORT == SSJ_TRANSPORT_BRIDGE
#include <Arduino_RouterBridge.h>
#endif

namespace {

struct Channel {
  ssj::Lsm6dsox imu;
  uint8_t id;
  bool present;
  uint16_t seq;
  uint32_t lastUs;
  int16_t gyro[3];
  bool gyroFresh;
};

Channel channels[2] = {
    {ssj::Lsm6dsox(SSJ_JAW_BUS, SSJ_JAW_ADDRESS), ssj::SENSOR_JAW, false, 0, 0, {0, 0, 0}, false},
    {ssj::Lsm6dsox(SSJ_REF_BUS, SSJ_REF_ADDRESS), ssj::SENSOR_REF, false, 0, 0, {0, 0, 0}, false},
};

struct Counters {
  uint32_t frames;
  uint32_t fifoOverruns;
  uint32_t txDropped;
  uint32_t loopMaxUs;
} counters = {0, 0, 0, 0};

bool streaming = true;
uint32_t lastInfoMs = 0;
uint32_t lastStatusMs = 0;
uint32_t lastLoopUs = 0;
const uint32_t samplePeriodUs = static_cast<uint32_t>(1000000.0f / ssj::Lsm6dsox::odrHz(SSJ_ODR_CODE));

// ------------------------------------------------------------------ transport
#if SSJ_TRANSPORT == SSJ_TRANSPORT_BRIDGE

// Frames wait in a ring until the Linux side collects them.
uint8_t queueBytes[SSJ_QUEUE_BYTES];
size_t queueHead = 0;  // next byte to read
size_t queueSize = 0;

size_t queueFree() { return SSJ_QUEUE_BYTES - queueSize; }

void emit(const uint8_t* frame, size_t n) {
  if (n > queueFree()) {
    counters.txDropped++;
    return;
  }
  size_t tail = (queueHead + queueSize) % SSJ_QUEUE_BYTES;
  for (size_t i = 0; i < n; i++) {
    queueBytes[tail] = frame[i];
    tail = (tail + 1) % SSJ_QUEUE_BYTES;
  }
  queueSize += n;
}

bool transportHasRoom() { return queueFree() > 16 * ssj::MAX_FRAME; }

#else

void emit(const uint8_t* frame, size_t n) { SSJ_SERIAL.write(frame, n); }

bool transportHasRoom() { return true; }

#endif

void sendFrame(uint8_t type, const uint8_t* body, size_t n) {
  uint8_t frame[ssj::MAX_FRAME];
  const size_t length = ssj::buildFrame(type, body, n, frame);
  if (length == 0) return;
  emit(frame, length);
  counters.frames++;
}

void sendSample(Channel& ch, uint32_t tUs, const int16_t accel[3], const int16_t gyro[3], uint8_t flags) {
  if (ch.seq != 0 && static_cast<int32_t>(tUs - ch.lastUs) <= 0) tUs = ch.lastUs + 1;  // keep time strictly increasing
  ch.lastUs = tUs;
  uint8_t body[ssj::SAMPLE_BODY];
  ssj::packSample(body, ch.id, ch.seq++, tUs, accel, gyro, flags);
  sendFrame(ssj::FRAME_SAMPLE, body, ssj::SAMPLE_BODY);
}

void sendInfo() {
  uint8_t body[ssj::INFO_BODY];
  const uint8_t sensors = static_cast<uint8_t>((channels[0].present ? 1 : 0) | (channels[1].present ? 2 : 0));
  ssj::packInfo(body, SSJ_FW_MAJOR, SSJ_FW_MINOR, SSJ_FW_PATCH, SSJ_USE_FIFO ? ssj::MODE_FIFO : ssj::MODE_POLLED, sensors,
                ssj::Lsm6dsox::odrHz(SSJ_ODR_CODE), ssj::Lsm6dsox::ACCEL_G_PER_LSB, ssj::Lsm6dsox::GYRO_DPS_PER_LSB);
  sendFrame(ssj::FRAME_INFO, body, ssj::INFO_BODY);
}

void sendStatus() {
  uint8_t body[ssj::STATUS_BODY];
  ssj::packStatus(body, millis(), counters.frames, channels[0].imu.errors() + channels[1].imu.errors(),
                  counters.fifoOverruns, counters.txDropped, counters.loopMaxUs);
  sendFrame(ssj::FRAME_STATUS, body, ssj::STATUS_BODY);
  counters.loopMaxUs = 0;
}

void sendLog(const char* text) {
  size_t n = strlen(text);
  if (n > ssj::MAX_BODY) n = ssj::MAX_BODY;
  sendFrame(ssj::FRAME_LOG, reinterpret_cast<const uint8_t*>(text), n);
}

// ---------------------------------------------------------------- acquisition
#if SSJ_USE_FIFO

// Drain what the sensor has batched. Gyro and accel words of one sample arrive
// back to back; a sample is emitted when its accel word arrives.
void service(Channel& ch) {
  bool overrun = false;
  const int level = ch.imu.fifoLevel(overrun);
  if (level <= 0) return;
  if (overrun) counters.fifoOverruns++;
  const uint32_t now = micros();
  const int backlog = level / 2;  // samples waiting, oldest first
  const int words = level < SSJ_FIFO_WORDS_PER_PASS ? level : SSJ_FIFO_WORDS_PER_PASS;
  int sample = 0;
  for (int i = 0; i < words; i++) {
    uint8_t tag;
    int16_t xyz[3];
    if (!ch.imu.fifoRead(tag, xyz)) return;
    if (tag == ssj::Lsm6dsox::TAG_GYRO) {
      memcpy(ch.gyro, xyz, sizeof(xyz));
      ch.gyroFresh = true;
    } else if (tag == ssj::Lsm6dsox::TAG_ACCEL) {
      // The newest sample in the FIFO was taken about now; older ones one period apart.
      const int age = backlog - 1 - sample;
      const uint32_t tUs = now - static_cast<uint32_t>(age > 0 ? age : 0) * samplePeriodUs;
      uint8_t flags = ssj::FLAG_ACCEL_FRESH;
      if (ch.gyroFresh) flags |= ssj::FLAG_GYRO_FRESH;
      if (overrun) flags |= ssj::FLAG_FIFO_OVERRUN;
      sendSample(ch, tUs, xyz, ch.gyro, flags);
      ch.gyroFresh = false;
      sample++;
    }
  }
}

#else

// Emit a sample whenever the sensor reports new data.
void service(Channel& ch) {
  uint8_t status = 0;
  if (!ch.imu.readStatus(status) || (status & 0x03) == 0) return;
  int16_t accel[3];
  int16_t gyro[3];
  const uint32_t tUs = micros();
  if (!ch.imu.readSample(accel, gyro)) return;
  uint8_t flags = status & 0x03;
  if (ch.seq != 0 && (tUs - ch.lastUs) > 2 * samplePeriodUs) flags |= ssj::FLAG_LATE;
  sendSample(ch, tUs, accel, gyro, flags);
}

#endif

// -------------------------------------------------------------------- bridge
#if SSJ_TRANSPORT == SSJ_TRANSPORT_BRIDGE

// Hands the Linux side whole frames, base64-encoded (the RPC carries text).
String getImuBatch() {
  static uint8_t raw[SSJ_BRIDGE_MAX_RAW];
  static char text[4 * ((SSJ_BRIDGE_MAX_RAW + 2) / 3) + 1];
  size_t take = 0;
  size_t scan = 0;
  while (scan < queueSize && scan < SSJ_BRIDGE_MAX_RAW) {
    if (queueBytes[(queueHead + scan) % SSJ_QUEUE_BYTES] == 0) take = scan + 1;  // end of a complete frame
    scan++;
  }
  for (size_t i = 0; i < take; i++) raw[i] = queueBytes[(queueHead + i) % SSJ_QUEUE_BYTES];
  queueHead = (queueHead + take) % SSJ_QUEUE_BYTES;
  queueSize -= take;
  ssj::base64Encode(raw, take, text);
  return String(text);
}

String getImuStatus() {
  sendInfo();
  String status = "sensors_ready=";
  status += (channels[0].present && channels[1].present) ? "1" : "0";
  status += ",jaw=";
  status += channels[0].present ? "ok" : "missing";
  status += ",ref=";
  status += channels[1].present ? "ok" : "missing";
  status += SSJ_USE_FIFO ? ",mode=fifo" : ",mode=polled";
  status += ",odr_hz=";
  status += String(ssj::Lsm6dsox::odrHz(SSJ_ODR_CODE), 0);
  status += ",frames=";
  status += String(counters.frames);
  status += ",dropped=";
  status += String(counters.txDropped);
  status += ",i2c_errors=";
  status += String(channels[0].imu.errors() + channels[1].imu.errors());
  return status;
}

#else

void handleCommands() {
  while (SSJ_SERIAL.available() > 0) {
    switch (SSJ_SERIAL.read()) {
      case 'i': sendInfo(); break;
      case 's': streaming = true; break;
      case 'p': streaming = false; break;
      case 'r': counters = {0, 0, 0, 0}; break;
      default: break;
    }
  }
}

#endif

}  // namespace

void setup() {
#if SSJ_TRANSPORT == SSJ_TRANSPORT_BRIDGE
  Bridge.begin();
#else
  SSJ_SERIAL.begin(SSJ_BAUD);
  const uint32_t waitStart = millis();
  while (!SSJ_SERIAL && millis() - waitStart < 2000) delay(10);
#endif

  SSJ_JAW_BUS.begin();
  SSJ_JAW_BUS.setClock(SSJ_I2C_HZ);
  SSJ_REF_BUS.begin();
  SSJ_REF_BUS.setClock(SSJ_I2C_HZ);

  for (Channel& ch : channels) ch.present = ch.imu.begin(SSJ_ODR_CODE, SSJ_USE_FIFO);

#if SSJ_TRANSPORT == SSJ_TRANSPORT_BRIDGE
  Bridge.provide_safe("get_imu_batch", getImuBatch);
  Bridge.provide_safe("get_imu_status", getImuStatus);
#endif

  sendInfo();
  if (!channels[0].present) sendLog("jaw IMU not found");
  if (!channels[1].present) sendLog("reference IMU not found");
  lastLoopUs = micros();
}

void loop() {
#if SSJ_TRANSPORT != SSJ_TRANSPORT_BRIDGE
  handleCommands();
#endif

  if (streaming && transportHasRoom()) {
    for (Channel& ch : channels) {
      if (ch.present) service(ch);
    }
  }

  const uint32_t nowMs = millis();
  if (nowMs - lastInfoMs >= 2000) {
    lastInfoMs = nowMs;
    sendInfo();
  }
  if (nowMs - lastStatusMs >= 1000) {
    lastStatusMs = nowMs;
    sendStatus();
  }

  const uint32_t nowUs = micros();
  const uint32_t elapsed = nowUs - lastLoopUs;
  if (elapsed > counters.loopMaxUs) counters.loopMaxUs = elapsed;
  lastLoopUs = nowUs;
}
