from __future__ import annotations

import numpy as np

from ssj.dsp.resample import DualResampler, StreamResampler, TimestampSmoother


def irregular_times(n: int, rate: float, rng: np.random.Generator, jitter: float = 0.3) -> np.ndarray:
    return 2.0 + np.cumsum((1.0 + jitter * rng.uniform(-1, 1, n)) / rate)


def test_linear_signal_is_reproduced_exactly():
    rng = np.random.default_rng(0)
    t = irregular_times(500, 45.0, rng)  # the rate and jitter of a slow transport
    x = np.stack([3.0 * t - 1.0, -0.5 * t], axis=1)
    k0, out, valid = StreamResampler(100.0).push(t, x)
    grid = (k0 + np.arange(len(out))) / 100.0
    assert grid[0] >= t[0] and grid[-1] <= t[-1]
    assert np.allclose(out[:, 0], 3.0 * grid - 1.0, atol=1e-9)
    assert np.allclose(out[:, 1], -0.5 * grid, atol=1e-9)
    assert valid.all()


def test_chunking_does_not_change_the_output():
    rng = np.random.default_rng(1)
    t = irregular_times(900, 208.0, rng)
    x = rng.normal(size=(900, 6))
    k_whole, whole, valid_whole = StreamResampler(200.0).push(t, x)
    resampler = StreamResampler(200.0)
    parts, valid_parts, cursor, first = [], [], 0, None
    while cursor < len(t):
        step = int(rng.integers(1, 30))
        k0, out, valid = resampler.push(t[cursor : cursor + step], x[cursor : cursor + step])
        if len(out) and first is None:
            first = k0
        parts.append(out)
        valid_parts.append(valid)
        cursor += step
    assert first == k_whole
    assert np.array_equal(np.concatenate(parts), whole)
    assert np.array_equal(np.concatenate(valid_parts), valid_whole)


def test_gaps_are_flagged_and_bad_timestamps_dropped():
    t = np.concatenate([np.arange(0, 1, 0.01), np.arange(1.5, 2.5, 0.01)]) + 10.0
    x = np.ones((len(t), 1))
    t_bad = np.insert(t, 20, t[19])  # a duplicate timestamp
    x_bad = np.insert(x, 20, 99.0, axis=0)
    k0, out, valid = StreamResampler(100.0, max_gap_s=0.1).push(t_bad, x_bad)
    grid = (k0 + np.arange(len(out))) / 100.0
    inside_gap = (grid > 10.995) & (grid < 11.495)
    assert not valid[inside_gap].any()
    assert valid[~inside_gap].all()
    assert np.allclose(out, 1.0)  # the duplicate did not leak in


def test_two_free_running_sensors_are_paired_on_one_grid():
    rng = np.random.default_rng(2)
    t_jaw = irregular_times(2000, 209.3, rng, jitter=0.05)
    t_ref = irregular_times(2000, 206.1, rng, jitter=0.05) + 0.37  # starts later, runs slower
    resampler = DualResampler(200.0)

    def signal(t):
        return np.stack([np.sin(2 * np.pi * 3 * t)] * 3, axis=1)

    for a in range(0, 2000, 137):
        resampler.push(0, t_jaw[a : a + 137], signal(t_jaw[a : a + 137]), 2 * signal(t_jaw[a : a + 137]))
        resampler.push(1, t_ref[a : a + 137], 3 * signal(t_ref[a : a + 137]), 4 * signal(t_ref[a : a + 137]))
    k0, rows, valid = resampler.pop()
    grid = (k0 + np.arange(len(rows))) / 200.0
    assert rows.shape[1] == 12 and valid.all()
    assert grid[0] >= max(t_jaw[0], t_ref[0]) and grid[-1] <= min(t_jaw[-1], t_ref[-1])
    expected = np.sin(2 * np.pi * 3 * grid)
    for column, gain in ((0, 1), (3, 2), (6, 3), (9, 4)):  # both sensors agree on the common time base
        assert np.max(np.abs(rows[:, column] - gain * expected)) < 0.02 * gain
    assert len(resampler.pop()[1]) == 0  # nothing is returned twice


def test_timestamp_smoother_recovers_the_sensor_clock():
    rng = np.random.default_rng(3)
    true_period = 1 / 208.0 * 1.012  # the oscillator is 1.2 % off nominal
    seq = np.delete(np.arange(4000), [700, 701, 702, 2500])  # a few lost frames
    true = 5.0 + seq * true_period
    stamped = np.ceil(true / 0.004) * 0.004 + rng.uniform(0, 0.0005, len(seq))  # stamped when the FIFO was drained
    smoother = TimestampSmoother(1 / 208.0)
    out = np.concatenate([smoother(seq[a : a + 97], stamped[a : a + 97]) for a in range(0, len(seq), 97)])
    raw_jitter = np.std(np.diff(stamped) - np.diff(true))
    smooth_jitter = np.std((np.diff(out) - np.diff(true))[1000:])
    assert smooth_jitter < 0.1 * raw_jitter
    assert abs(smoother.period - true_period) / true_period < 2e-3
    assert np.all(np.diff(out) > 0)
