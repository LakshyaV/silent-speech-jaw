# Wire protocol

The streamer firmware and the host exchange a byte stream of self-delimiting
binary frames. The same stream is used on every transport: a serial port, the
Arduino UNO Q RPC bridge (base64-wrapped in batches), and the TCP relay.

Reference implementations: [`firmware/ssj_streamer/framing.h`](../firmware/ssj_streamer/framing.h)
and [`src/ssj/io/protocol.py`](../src/ssj/io/protocol.py). The test-suite compiles
the former natively and decodes its output with the latter.

## Framing

```
frame   = COBS( type | body | crc_lo | crc_hi )  0x00
```

- **COBS** removes zero bytes from the payload, so `0x00` only ever appears as
  the frame delimiter. After any corruption the receiver resynchronises at the
  next zero byte.
- **CRC** is CRC-16/CCITT-FALSE (polynomial `0x1021`, initial value `0xFFFF`, no
  reflection) over `type | body`, least significant byte first. The check value
  for `"123456789"` is `0x29B1`.
- All multi-byte fields are little-endian.

## Frame types

### `0x01` SAMPLE (20-byte body, 25 bytes on the wire)

| Offset | Field | Type | Meaning |
| --- | --- | --- | --- |
| 0 | `sensor` | u8 | `0` jaw, `1` reference |
| 1 | `seq` | u16 | per-sensor sample counter (wraps) |
| 3 | `t_us` | u32 | MCU `micros()` at which the sample was taken (wraps after 71.6 min) |
| 7 | `accel` | 3 x i16 | accelerometer x, y, z in LSB |
| 13 | `gyro` | 3 x i16 | gyroscope x, y, z in LSB |
| 19 | `flags` | u8 | see below |

One frame carries one reading of one sensor. The two IMUs run on their own
oscillators, so they are not paired on the wire; the host resamples each stream
onto a common time grid.

| Flag bit | Name | Meaning |
| --- | --- | --- |
| 0 | `ACCEL_FRESH` | the accelerometer value is new |
| 1 | `GYRO_FRESH` | the gyroscope value is new |
| 2 | `READ_ERROR` | an I2C error occurred for this sample |
| 3 | `FIFO_OVERRUN` | the sensor FIFO overflowed before this sample: earlier samples were lost without being numbered. The host restarts its clock tracker here, so the hole appears as a gap |
| 4 | `LATE` | polled mode: more than two sample periods since the previous sample |

### `0x02` INFO (18-byte body)

| Offset | Field | Type |
| --- | --- | --- |
| 0 | `proto` | u8, protocol version (`1`) |
| 1 | `fw_major`, `fw_minor`, `fw_patch` | 3 x u8 |
| 4 | `mode` | u8: `0` polled, `1` FIFO |
| 5 | `sensors` | u8 bit mask of detected sensors (bit 0 jaw, bit 1 reference) |
| 6 | `odr_hz` | f32, nominal output data rate |
| 10 | `accel_g_per_lsb` | f32 |
| 14 | `gyro_dps_per_lsb` | f32 |

Sent at boot, every two seconds, and on request. Samples are raw LSBs; these
scale factors are the only place units are defined, so changing a sensor range
in the firmware cannot silently change what the host computes.

### `0x03` STATUS (24-byte body)

Six u32 counters, sent once per second: `uptime_ms`, `frames`, `i2c_errors`,
`fifo_overruns`, `tx_dropped`, `loop_max_us` (longest main-loop pass since the
previous STATUS frame).

### `0x04` LOG

UTF-8 text, at most 96 bytes. Used for start-up diagnostics such as a sensor
that did not answer.

## Host commands (serial transport)

Single bytes: `i` send an INFO frame, `s` start streaming, `p` pause, `r` reset
counters.

## Time

`seq` and `t_us` are unwrapped to 64 bits by the receiver. In FIFO mode the
sensor's own clock spaces the samples and the MCU can only stamp them when it
drains the FIFO, so the host recovers the underlying uniform clock from
(sequence number, timestamp) pairs with an alpha-beta tracker
(`ssj.dsp.resample.TimestampSmoother`). Every later stage works on the resulting
time axis, never on sample counts.

## Arduino UNO Q bridge

On the UNO Q the MCU cannot reach a host directly: its `Serial` is a text
console tunnelled through the RPC bridge. The firmware therefore queues frames
and exposes two RPC methods to the board's Linux side:

| Method | Returns |
| --- | --- |
| `get_imu_batch` | pending whole frames, base64-encoded, at most `SSJ_BRIDGE_MAX_RAW` raw bytes per call |
| `get_imu_status` | a human-readable status line; also queues a fresh INFO frame |

The Linux side decodes the batches (`BridgeSource`) or forwards the raw bytes
over TCP to a laptop (`TcpSource`, `--source tcp:<board>:5555`).
