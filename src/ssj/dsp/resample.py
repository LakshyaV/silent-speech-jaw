"""Timestamp-driven resampling of the two IMU streams onto one uniform grid.

Fixed-rate assumptions do not survive contact with hardware: the transport
jitters, frames get lost, and the two IMUs run on independent oscillators that
drift apart by a sample every second or so. Everything downstream therefore
works on a grid defined by *time* (``t = k / fs`` in the device clock), and each
sensor is interpolated onto it from its own timestamps.

All classes here are streaming: feeding a recording in one call or in many
small chunks produces identical output.
"""

from __future__ import annotations

import math

import numpy as np


class TimestampSmoother:
    """Regularises timestamps of a sensor-clocked stream.

    In FIFO mode the sensor's own oscillator spaces the samples perfectly, but
    the MCU can only stamp them when it drains the FIFO, which adds up to one
    sample period of jitter. An alpha-beta tracker over (sequence number, time)
    recovers the underlying uniform clock, including its true rate, and stays
    correct across lost frames because it advances by sequence number.

    Sequence numbers only cover what the board sent. When the sensor's FIFO
    overruns, samples are lost before they are numbered and the board flags the
    ones it drains next; passing those as ``restart`` re-anchors the clock on
    their stamps instead of carrying it across the hole.
    """

    def __init__(self, nominal_period_s: float, alpha: float = 0.02, beta: float = 2e-4):
        self.period = float(nominal_period_s)
        self.alpha = alpha
        self.beta = beta
        self._seq: int | None = None
        self._time = 0.0

    def __call__(self, seq: np.ndarray, t: np.ndarray, restart: np.ndarray | None = None) -> np.ndarray:
        out = np.empty(len(t), dtype=np.float64)
        last_seq, time_, period = self._seq, self._time, self.period
        alpha, beta = self.alpha, self.beta
        for i in range(len(t)):
            s = int(seq[i])
            observed = float(t[i])
            if last_seq is None or s <= last_seq or s - last_seq > 2000 or (restart is not None and restart[i]):
                time_ = observed  # first sample, a restarted stream, or samples lost before numbering
            else:
                step = s - last_seq
                time_ += period * step
                residual = observed - time_
                if abs(residual) > 0.25:  # the clocks disagree outright: resynchronise
                    time_ = observed
                else:
                    time_ += alpha * residual
                    period += beta * residual / step
            last_seq = s
            out[i] = time_
        self._seq, self._time, self.period = last_seq, time_, period
        return out


class StreamResampler:
    """Linear interpolation of one irregular multi-channel stream onto ``k / fs``."""

    def __init__(self, fs: float, max_gap_s: float = 0.1):
        self.fs = float(fs)
        self.max_gap_s = float(max_gap_s)
        self._t_prev: float | None = None
        self._x_prev: np.ndarray | None = None
        self._next_k: int | None = None

    def push(self, t: np.ndarray, x: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
        """Add samples; return ``(k0, values[n, C], valid[n])`` for the new grid points.

        ``valid`` is False where the grid point was bridged across a gap longer
        than ``max_gap_s`` (lost frames), so downstream code can mask it.
        """
        t = np.asarray(t, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        n_channels = x.shape[1] if x.ndim == 2 else 0
        empty = (self._next_k or 0, np.zeros((0, n_channels)), np.zeros(0, dtype=bool))
        if len(t) == 0:
            return empty

        # Keep strictly increasing timestamps only (duplicates and reordering happen).
        floor = self._t_prev if self._t_prev is not None else -math.inf
        running = np.maximum.accumulate(np.concatenate(([floor], t)))[:-1]
        keep = t > running
        t, x = t[keep], x[keep]
        if len(t) == 0:
            return empty

        if self._t_prev is None:
            self._next_k = math.ceil(t[0] * self.fs - 1e-9)
            times, values = t, x
        else:
            times = np.concatenate(([self._t_prev], t))
            values = np.concatenate((self._x_prev[None, :], x), axis=0)
        self._t_prev, self._x_prev = float(t[-1]), x[-1].copy()

        last_k = math.floor(times[-1] * self.fs + 1e-9)
        k0 = int(self._next_k)
        if last_k < k0 or len(times) < 2:
            return k0, np.zeros((0, values.shape[1])), np.zeros(0, dtype=bool)

        grid = np.arange(k0, last_k + 1, dtype=np.float64) / self.fs
        right = np.clip(np.searchsorted(times, grid, side="left"), 1, len(times) - 1)
        left = right - 1
        span = times[right] - times[left]
        weight = np.clip((grid - times[left]) / span, 0.0, 1.0)[:, None]
        out = values[left] * (1.0 - weight) + values[right] * weight
        self._next_k = last_k + 1
        # A grid point that falls on a gap is not trustworthy unless it coincides with a sample.
        on_sample = (weight[:, 0] < 1e-9) | (weight[:, 0] > 1.0 - 1e-9)
        return k0, out, (span <= self.max_gap_s) | on_sample


class DualResampler:
    """Pairs the jaw and reference streams on a shared uniform grid.

    Output rows are ``[jaw_accel(3), jaw_gyro(3), ref_accel(3), ref_gyro(3)]``
    in physical units (g, deg/s) for consecutive grid indices.
    """

    N_CHANNELS = 12

    def __init__(self, fs: float, max_gap_s: float = 0.1):
        self.fs = float(fs)
        self._resamplers = (StreamResampler(fs, max_gap_s), StreamResampler(fs, max_gap_s))
        self._k0: list[int | None] = [None, None]
        self._values = [np.zeros((0, 6)), np.zeros((0, 6))]
        self._valid = [np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)]

    def push(self, sensor: int, t: np.ndarray, accel: np.ndarray, gyro: np.ndarray) -> None:
        if len(t) == 0:
            return
        k0, values, valid = self._resamplers[sensor].push(t, np.concatenate((accel, gyro), axis=1))
        if len(values) == 0:
            return
        if self._k0[sensor] is None:
            self._k0[sensor] = k0
        self._values[sensor] = np.concatenate((self._values[sensor], values), axis=0)
        self._valid[sensor] = np.concatenate((self._valid[sensor], valid))

    def backlog_s(self) -> float:
        """Seconds of data waiting for the other sensor (large values mean one IMU stalled)."""
        return abs(len(self._values[0]) - len(self._values[1])) / self.fs

    def pop(self) -> tuple[int, np.ndarray, np.ndarray]:
        """Return ``(k0, rows[n, 12], valid[n])`` for all grid points both sensors cover."""
        if self._k0[0] is None or self._k0[1] is None:
            return 0, np.zeros((0, self.N_CHANNELS)), np.zeros(0, dtype=bool)
        start = max(self._k0[0], self._k0[1])
        for s in (0, 1):  # drop the head of whichever sensor started earlier
            skip = start - self._k0[s]
            if skip > 0:
                self._values[s] = self._values[s][skip:]
                self._valid[s] = self._valid[s][skip:]
                self._k0[s] = start
        n = min(len(self._values[0]), len(self._values[1]))
        if n == 0:
            return start, np.zeros((0, self.N_CHANNELS)), np.zeros(0, dtype=bool)
        rows = np.concatenate((self._values[0][:n], self._values[1][:n]), axis=1)
        valid = self._valid[0][:n] & self._valid[1][:n]
        for s in (0, 1):
            self._values[s] = self._values[s][n:]
            self._valid[s] = self._valid[s][n:]
            self._k0[s] = start + n
        return start, rows, valid
