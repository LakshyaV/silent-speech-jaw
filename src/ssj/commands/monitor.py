"""Watch the sensor link: delivered sample rate, timestamp jitter, loss and device counters.

    ssj monitor
    ssj monitor --source serial:/dev/cu.usbmodem1101 --seconds 30
    ssj monitor --source replay:data/raw/s01/<session>   # stream health of a recording

Run this before the first recording with new hardware or firmware settings.
Both sensors should report close to the nominal rate with no loss.
"""

from __future__ import annotations

import argparse
import time

NAME = "monitor"
HELP = "show live stream health for both IMUs"
ORDER = 22


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", default="serial", help="serial[:port], tcp:<host>[:port] or replay:<session dir>")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--interval", type=float, default=1.0)


def run(args: argparse.Namespace) -> int:
    import numpy as np

    from ..dsp.activity import session_health
    from ..io.protocol import SENSOR_NAMES, split_sensors
    from ..io.sources import open_source

    kwargs = {"baud": args.baud} if args.source.startswith("serial") else {}
    with open_source(args.source, **kwargs) as source:
        info = source.wait_info()
        print(
            f"firmware {'.'.join(map(str, info.firmware))} | {info.mode_name} mode | nominal {info.odr_hz:.0f} Hz | "
            f"accel {info.accel_g_per_lsb * 1e3:.3f} mg/LSB | gyro {info.gyro_dps_per_lsb * 1e3:.2f} mdps/LSB"
        )

        def report(window: list[np.ndarray]) -> list[np.ndarray]:
            """Print one health line and return the last sample of each sensor.

            Those samples open the next window, so the interval across a report
            boundary is counted and a hole that straddles two reports is seen.
            """
            parts = split_sensors(np.concatenate(window))
            line = []
            for sensor, part in parts.items():
                health = session_health(part["t_us"], part["seq"], info.odr_hz)
                line.append(
                    f"{SENSOR_NAMES[sensor]}: {health['rate_hz']:6.1f} Hz  jitter {health['jitter_ms']:4.1f} ms  "
                    f"gap {health['max_gap_ms']:5.1f} ms  loss {100 * health['loss']:4.1f}%"
                )
            status = getattr(source, "status", None)
            if status is not None:
                line.append(
                    f"i2c errors {status.i2c_errors}, fifo overruns {status.fifo_overruns}, tx dropped {status.tx_dropped}"
                )
            print(" | ".join(line), flush=True)
            return [part[-1:] for part in parts.values() if len(part)]

        deadline = time.monotonic() + args.seconds
        window: list[np.ndarray] = []
        fresh, reported = 0, False
        tick = time.monotonic() + args.interval
        while time.monotonic() < deadline and not source.exhausted:
            rows = source.read()
            if len(rows):
                window.append(rows)
                fresh += len(rows)
            if fresh and time.monotonic() >= tick:
                window, fresh, reported = report(window), 0, True
                tick = time.monotonic() + args.interval
        # A stream that ends between two reports (a replayed session, a relay that
        # closes) is summarised from what arrived since the last one.
        if fresh and (source.exhausted or not reported):
            report(window)
        stats = getattr(source, "stats", None)
        if stats is not None:
            print(f"frames ok {stats.frames_ok}, crc errors {stats.crc_errors}, framing errors {stats.framing_errors}")
    return 0
