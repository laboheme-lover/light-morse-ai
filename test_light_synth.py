import numpy as np
import pytest

from light_synth import CONDITIONS, SynthParams, add_conditions, on_intervals, render, synthesize_char


def runs(binary):
    out, start = [], 0
    for i in range(1, len(binary) + 1):
        if i == len(binary) or binary[i] != binary[start]:
            out.append((bool(binary[start]), i - start))
            start = i
    return out


def clean_params(unit=0.3):
    return SynthParams(unit=unit, fps=30.0, exposure=0.5, phase=0.0, off_level=13.5, on_level=244.0)


def test_intervals_follow_unit_lengths_without_jitter():
    rng = np.random.default_rng(0)
    intervals, end = on_intervals(["-.-.."], 0.3, rng, jitter=0.0)  # キ
    lengths = [round(b - a, 6) for a, b in intervals]
    gaps = [round(b[0] - a[1], 6) for a, b in zip(intervals, intervals[1:])]
    assert lengths == [0.9, 0.3, 0.9, 0.3, 0.3]
    assert gaps == [0.3] * 4
    assert end == pytest.approx(0.9 * 2 + 0.3 * 3 + 0.3 * 4)


def test_dakuten_char_has_three_unit_gap_between_codes():
    rng = np.random.default_rng(0)
    intervals, _ = on_intervals([".-..", ".."], 0.3, rng, jitter=0.0)  # ガ = カ + 濁点
    gaps = [round(b[0] - a[1], 6) for a, b in zip(intervals, intervals[1:])]
    assert gaps == [0.3, 0.3, 0.3, 0.9, 0.3]


def test_rendered_frames_show_dot_and_dash_lengths():
    rng = np.random.default_rng(0)
    intervals, end = on_intervals(["-.-.."], 0.3, rng, jitter=0.0, start=0.5)
    _, v = render(intervals, end + 0.5, clean_params())
    on = [n for is_on, n in runs(v > (v.max() + v.min()) / 2) if is_on]
    assert len(on) == 5
    assert all(8 <= n <= 10 for n in (on[1], on[3], on[4]))  # 短点 0.3秒 ≒ 9コマ
    assert all(26 <= n <= 28 for n in (on[0], on[2]))  # 長点 0.9秒 ≒ 27コマ
    assert v.min() == pytest.approx(13.5, abs=0.5) and v.max() == pytest.approx(244, abs=1)


def test_synthesize_char_adds_lead_and_trail_off_frames():
    rng = np.random.default_rng(1)
    v, p = synthesize_char("ヘ", rng)
    b = v > (v.max() + v.min()) / 2
    assert not b[0] and not b[-1] and b.any()


@pytest.mark.parametrize("cond", [(c,) for c in CONDITIONS] + [CONDITIONS])
def test_conditions_keep_range_and_length(cond):
    rng = np.random.default_rng(2)
    v, _ = synthesize_char("ア", rng)
    out = add_conditions(v, cond, rng)
    assert out.shape == v.shape
    assert out.min() >= 0 and out.max() <= 255
    assert not np.allclose(out, v)


def test_dusk_and_fog_reduce_contrast():
    rng = np.random.default_rng(3)
    v, _ = synthesize_char("ア", rng)
    base = np.percentile(v, 98) - np.percentile(v, 2)
    for cond in ("dusk", "fog"):
        out = add_conditions(v, (cond,), np.random.default_rng(4))
        assert np.percentile(out, 98) - np.percentile(out, 2) < base * 0.8, cond
