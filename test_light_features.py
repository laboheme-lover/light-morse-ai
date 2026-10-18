import numpy as np

from light_features import INPUT_LENGTH, SAMPLES_PER_UNIT, normalize, prepare_input
from light_synth import SynthParams, on_intervals, render


def trace_for(code, unit, fps=30.0, dusk=False):
    rng = np.random.default_rng(0)
    intervals, end = on_intervals([code], unit, rng, jitter=0.0, start=0.6)
    p = SynthParams(unit=unit, fps=fps, exposure=0.6, phase=0.3, off_level=13.5, on_level=244.0)
    _, v = render(intervals, end + 0.6, p)
    return v * 0.2 + 120 if dusk else v


def test_normalize_maps_off_on_to_0_1():
    x = normalize(trace_for("-.-..", 0.3))
    assert x.min() == 0.0 and x.max() == 1.0


def test_same_char_looks_same_at_different_speeds():
    a = prepare_input(trace_for("-.-..", 0.3), 0.3 * 30)
    b = prepare_input(trace_for("-.-..", 0.5), 0.5 * 30)
    assert a.shape == (INPUT_LENGTH,) and a.dtype == np.float32
    assert np.mean(np.abs(a - b)) < 0.05


def test_low_contrast_gives_same_input():
    a = prepare_input(trace_for("-.-..", 0.3), 9.0)
    b = prepare_input(trace_for("-.-..", 0.3, dusk=True), 9.0)
    assert np.mean(np.abs(a - b)) < 0.02


def test_signal_starts_one_unit_from_left():
    x = prepare_input(trace_for("-", 0.3), 9.0)
    first_on = int(np.argmax(x > 0.5))
    assert abs(first_on - SAMPLES_PER_UNIT) <= 2
    # 長点(3単位)のあとは消灯で埋まる
    assert x[first_on: first_on + 3 * SAMPLES_PER_UNIT - 2].min() > 0.5
    assert x[first_on + 3 * SAMPLES_PER_UNIT + 2:].max() < 0.5
