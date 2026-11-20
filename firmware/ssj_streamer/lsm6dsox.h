// Minimal register-level LSM6DSOX driver for high-rate dual-IMU streaming.
//
// The stock Arduino_LSM6DSOX library fixes the output data rate at 104 Hz and
// reads accelerometer and gyroscope in separate transactions. Phoneme-level
// sensing needs more: a configurable rate, raw LSBs (scaling is done on the
// host), one burst read per sample, and the sensor's own FIFO so that samples
// are spaced by the sensor clock and survive pauses in the main loop.
//
// Register map and bit fields follow the ST datasheet (DS12814) and ST's
// lsm6dsox_reg.h. The same registers exist on the LSM6DSO, LSM6DSR and
// ISM330DHCX.
#pragma once

#include <Arduino.h>
#include <Wire.h>

namespace ssj {

class Lsm6dsox {
 public:
  // Output data rate codes (CTRL1_XL / CTRL2_G / FIFO_CTRL3).
  static constexpr uint8_t ODR_52HZ = 3;
  static constexpr uint8_t ODR_104HZ = 4;
  static constexpr uint8_t ODR_208HZ = 5;
  static constexpr uint8_t ODR_417HZ = 6;

  // Full-scale ranges are fixed: +-4 g and +-500 dps cover speech and head motion
  // with headroom while keeping the resolution high.
  static constexpr float ACCEL_G_PER_LSB = 0.000122f;  // 0.122 mg/LSB at +-4 g
  static constexpr float GYRO_DPS_PER_LSB = 0.0175f;   // 17.5 mdps/LSB at +-500 dps

  static constexpr uint8_t TAG_GYRO = 0x01;
  static constexpr uint8_t TAG_ACCEL = 0x02;

  Lsm6dsox(TwoWire& bus, uint8_t address) : bus_(bus), address_(address) {}

  static float odrHz(uint8_t code) {
    switch (code) {
      case 1: return 12.5f;
      case 2: return 26.0f;
      case 3: return 52.0f;
      case 4: return 104.0f;
      case 5: return 208.0f;
      case 6: return 417.0f;
      case 7: return 833.0f;
      default: return 0.0f;
    }
  }

  // Probe, reset and configure. Returns false if no supported device answers.
  bool begin(uint8_t odrCode, bool useFifo) {
    uint8_t id = 0;
    if (!readN(REG_WHO_AM_I, &id, 1)) return false;
    whoAmI_ = id;
    const bool dsoFamily = (id == 0x6C || id == 0x6B);  // LSM6DSO(X) / LSM6DSR, ISM330DHCX
    if (!dsoFamily && (useFifo || (id != 0x69 && id != 0x6A))) return false;

    if (!write8(REG_CTRL3_C, 0x01)) return false;  // SW_RESET
    delay(20);
    if (!write8(REG_CTRL3_C, 0x44)) return false;  // BDU | IF_INC
    if (dsoFamily) {
      uint8_t ctrl9 = 0;
      if (readN(REG_CTRL9_XL, &ctrl9, 1)) write8(REG_CTRL9_XL, ctrl9 | 0x02);  // disable I3C on a plain I2C bus
    }
    const uint8_t odr = static_cast<uint8_t>(odrCode << 4);
    if (!write8(REG_CTRL1_XL, odr | 0x08)) return false;  // FS_XL = +-4 g
    if (!write8(REG_CTRL2_G, odr | 0x04)) return false;   // FS_G = +-500 dps
    if (useFifo) {
      if (!write8(REG_FIFO_CTRL4, 0x00)) return false;  // bypass: empties the FIFO
      write8(REG_FIFO_CTRL1, 0x00);
      write8(REG_FIFO_CTRL2, 0x00);
      if (!write8(REG_FIFO_CTRL3, static_cast<uint8_t>((odrCode << 4) | odrCode))) return false;  // batch gyro + accel
      if (!write8(REG_FIFO_CTRL4, 0x06)) return false;  // continuous mode, no timestamp/temperature batching
    }
    delay(30);  // let the first samples settle
    return true;
  }

  // Polled mode: bit 0 = new accel data, bit 1 = new gyro data.
  bool readStatus(uint8_t& status) { return readN(REG_STATUS, &status, 1); }

  // Polled mode: one burst read of gyro (0x22..0x27) and accel (0x28..0x2D).
  bool readSample(int16_t accel[3], int16_t gyro[3]) {
    uint8_t raw[12];
    if (!readN(REG_OUTX_L_G, raw, sizeof(raw))) return false;
    for (uint8_t i = 0; i < 3; i++) {
      gyro[i] = static_cast<int16_t>(raw[2 * i] | (raw[2 * i + 1] << 8));
      accel[i] = static_cast<int16_t>(raw[6 + 2 * i] | (raw[6 + 2 * i + 1] << 8));
    }
    return true;
  }

  // FIFO mode: number of unread words (each word is one tagged 3-axis sample).
  // Returns -1 on a bus error. `overrun` reports that the FIFO overflowed.
  int fifoLevel(bool& overrun) {
    uint8_t status[2];
    if (!readN(REG_FIFO_STATUS1, status, 2)) return -1;
    overrun = (status[1] & 0x48) != 0;  // FIFO_OVR_IA | FIFO_OVR_LATCHED
    return status[0] | ((status[1] & 0x03) << 8);
  }

  // FIFO mode: pop one word. `tag` is the sensor tag (TAG_GYRO / TAG_ACCEL / other).
  bool fifoRead(uint8_t& tag, int16_t xyz[3]) {
    uint8_t raw[7];
    if (!readN(REG_FIFO_DATA_OUT_TAG, raw, sizeof(raw))) return false;
    tag = raw[0] >> 3;
    for (uint8_t i = 0; i < 3; i++) xyz[i] = static_cast<int16_t>(raw[1 + 2 * i] | (raw[2 + 2 * i] << 8));
    return true;
  }

  uint32_t errors() const { return errors_; }
  uint8_t whoAmI() const { return whoAmI_; }

 private:
  static constexpr uint8_t REG_FIFO_CTRL1 = 0x07;
  static constexpr uint8_t REG_FIFO_CTRL2 = 0x08;
  static constexpr uint8_t REG_FIFO_CTRL3 = 0x09;
  static constexpr uint8_t REG_FIFO_CTRL4 = 0x0A;
  static constexpr uint8_t REG_WHO_AM_I = 0x0F;
  static constexpr uint8_t REG_CTRL1_XL = 0x10;
  static constexpr uint8_t REG_CTRL2_G = 0x11;
  static constexpr uint8_t REG_CTRL3_C = 0x12;
  static constexpr uint8_t REG_CTRL9_XL = 0x18;
  static constexpr uint8_t REG_STATUS = 0x1E;
  static constexpr uint8_t REG_OUTX_L_G = 0x22;
  static constexpr uint8_t REG_FIFO_STATUS1 = 0x3A;
  static constexpr uint8_t REG_FIFO_DATA_OUT_TAG = 0x78;

  bool write8(uint8_t reg, uint8_t value) {
    bus_.beginTransmission(address_);
    bus_.write(reg);
    bus_.write(value);
    if (bus_.endTransmission() != 0) {
      errors_++;
      return false;
    }
    return true;
  }

  bool readN(uint8_t reg, uint8_t* out, size_t n) {
    bus_.beginTransmission(address_);
    bus_.write(reg);
    if (bus_.endTransmission(false) != 0) {
      errors_++;
      return false;
    }
    if (bus_.requestFrom(address_, n) != n) {
      errors_++;
      return false;
    }
    for (size_t i = 0; i < n; i++) out[i] = static_cast<uint8_t>(bus_.read());
    return true;
  }

  TwoWire& bus_;
  uint8_t address_;
  uint8_t whoAmI_ = 0;
  uint32_t errors_ = 0;
};

}  // namespace ssj
