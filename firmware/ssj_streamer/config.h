// Build-time configuration of the dual-IMU streamer.
//
// Everything a particular rig can differ in is here; the sketch itself does not
// need editing. See docs/hardware.md for wiring.
#pragma once

// ---------------------------------------------------------------- transport
// SERIAL: binary frames on a byte-transparent serial port (USB CDC or UART)
//         straight to a host computer.
// BRIDGE: Arduino UNO Q. On this board `Serial` is a text console tunnelled
//         through the RPC bridge and is neither fast nor binary-safe, so frames
//         are queued on the MCU and collected in batches by the board's Linux
//         side (see app/), which decodes them or relays them to a laptop.
#define SSJ_TRANSPORT_SERIAL 0
#define SSJ_TRANSPORT_BRIDGE 1
#ifndef SSJ_TRANSPORT
#if defined(ARDUINO_UNO_Q)
#define SSJ_TRANSPORT SSJ_TRANSPORT_BRIDGE
#else
#define SSJ_TRANSPORT SSJ_TRANSPORT_SERIAL
#endif
#endif

#ifndef SSJ_SERIAL
#define SSJ_SERIAL Serial
#endif
#ifndef SSJ_BAUD
#define SSJ_BAUD 921600
#endif

// -------------------------------------------------------------- acquisition
// FIFO (1): each IMU batches samples in its own FIFO at its own clock; the MCU
//           drains them. Sample spacing is exact and a slow loop loses nothing.
// POLLED (0): the MCU polls the data-ready flags and reads the output registers.
#ifndef SSJ_USE_FIFO
#define SSJ_USE_FIFO 1
#endif

// Output data rate code: 4 = 104 Hz, 5 = 208 Hz, 6 = 417 Hz.
#ifndef SSJ_ODR_CODE
#define SSJ_ODR_CODE 5
#endif

// ------------------------------------------------------------------- wiring
// Jaw IMU on the Qwiic connector, reference IMU on the header I2C pins. To put
// both on one bus instead, give the reference IMU the alternate address 0x6B
// (SDO/SA0 pulled high) and set SSJ_REF_BUS to the same bus as SSJ_JAW_BUS.
#ifndef SSJ_JAW_BUS
#define SSJ_JAW_BUS Wire1
#endif
#ifndef SSJ_JAW_ADDRESS
#define SSJ_JAW_ADDRESS 0x6A
#endif
#ifndef SSJ_REF_BUS
#define SSJ_REF_BUS Wire
#endif
#ifndef SSJ_REF_ADDRESS
#define SSJ_REF_ADDRESS 0x6A
#endif
#ifndef SSJ_I2C_HZ
#define SSJ_I2C_HZ 400000
#endif

// ------------------------------------------------------------------- tuning
// FIFO words drained from one sensor per loop pass (two words per sample).
#ifndef SSJ_FIFO_WORDS_PER_PASS
#define SSJ_FIFO_WORDS_PER_PASS 32
#endif
// Bridge transport: bytes of queued frames, and raw bytes returned per RPC call.
#ifndef SSJ_QUEUE_BYTES
#define SSJ_QUEUE_BYTES 8192
#endif
#ifndef SSJ_BRIDGE_MAX_RAW
#define SSJ_BRIDGE_MAX_RAW 360
#endif

#define SSJ_FW_MAJOR 0
#define SSJ_FW_MINOR 1
#define SSJ_FW_PATCH 0
