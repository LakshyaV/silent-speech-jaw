# Hardware and firmware

## Parts

| Part | Notes |
| --- | --- |
| Arduino UNO Q | STM32U585 MCU samples the sensors; the Qualcomm Linux side can run the whole recogniser. Other Arduino-compatible boards with two I2C buses (or one bus and two addresses) and a USB serial port work with the serial transport. |
| 2 x LSM6DSOX breakout | 6-axis IMU. Qwiic / STEMMA QT breakouts need no soldering. LSM6DSO, LSM6DSR and ISM330DHCX share the register map. |
| Qwiic cables, skin-safe tape | Thin, flexible cable matters more than anything else: a stiff cable moves the sensor. |

## Placement

- **Jaw IMU**: on the skin over the mandible, at the angle of the jaw or on the
  body of the mandible below the cheek. It must follow the bone, so place it
  where there is little soft tissue and tape it flat.
- **Reference IMU**: on the mastoid process, the bony bump directly behind the
  ear. It follows the skull and measures head motion.

Orientation does not matter. The calibration routine at the start of every
session finds each sensor's orientation from a rest, a nod-and-shake and an
open-and-close movement. Keep the placement roughly the same between sessions
and take the note field of `ssj record` seriously: write down side, position and
how the cable is routed.

## Wiring

Default configuration ([config.h](../firmware/ssj_streamer/config.h)):

| Sensor | Bus | Address |
| --- | --- | --- |
| jaw | `Wire1` (the Qwiic connector) | `0x6A` |
| reference | `Wire` (header SDA / SCL, 3.3 V) | `0x6A` |

To put both sensors on one bus, give the reference IMU the alternate address
`0x6B` (SDO/SA0 pulled high, usually a solder jumper) and build with
`-DSSJ_REF_BUS=Wire1 -DSSJ_REF_ADDRESS=0x6B`.

## Firmware

[`firmware/ssj_streamer`](../firmware/ssj_streamer) streams raw samples and does
nothing else. Options are compile-time defines:

| Define | Default | Meaning |
| --- | --- | --- |
| `SSJ_USE_FIFO` | `1` | `1`: each IMU batches samples in its FIFO at its own clock and the MCU drains them. `0`: the MCU polls the data-ready flags. |
| `SSJ_ODR_CODE` | `5` | output data rate: `4` = 104 Hz, `5` = 208 Hz, `6` = 417 Hz |
| `SSJ_TRANSPORT` | board-dependent | `0` serial, `1` UNO Q RPC bridge (default on the UNO Q) |
| `SSJ_BAUD` | `921600` | serial transport only |

FIFO mode is the default because it decouples sampling from everything else the
MCU does: sample spacing is set by the sensor's oscillator, and a slow main loop
or a busy transport delays samples without losing them.

### Arduino UNO Q

On the UNO Q, `Serial` is a text console tunnelled through the RPC bridge, not
a byte-transparent link to a host, so the firmware uses the bridge transport:
frames are queued on the MCU and collected by the board's Linux side.

Build check with the pinned versions:

```bash
arduino-cli compile --profile unoq firmware/ssj_streamer
```

To run it, package the App Lab application, which contains the sketch
([deployment](deployment.md)):

```bash
ssj app package --relay --out dist/ssj-relay      # record and decode on a laptop
ssj app package --bundle runs/base/bundle --out dist/ssj-app   # decode on the board
```

### Other boards (serial transport)

```bash
arduino-cli compile --fqbn arduino:renesas_uno:unor4wifi --upload -p /dev/cu.usbmodemXXXX firmware/ssj_streamer
```

## Checking the link

```bash
ssj monitor                                   # serial transport
ssj monitor --source tcp:<board-ip>:5555      # UNO Q relay
```

```
firmware 0.1.0 | fifo mode | nominal 208 Hz | accel 0.122 mg/LSB | gyro 17.50 mdps/LSB
jaw:  208.4 Hz  jitter  0.4 ms  gap   5.1 ms  loss  0.0% | ref:  207.1 Hz  jitter  0.5 ms  gap   5.2 ms  loss  0.0%
```

Both sensors should report close to the nominal rate with no loss. The two
rates differ slightly because each sensor has its own oscillator; the host
handles that. `i2c errors` above zero points to wiring; `fifo overruns` or
`tx dropped` mean the transport cannot keep up, in which case lower
`SSJ_ODR_CODE` to `4`.
