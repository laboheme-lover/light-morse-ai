import numpy as np

from light_sequence import binarize, estimate_unit_frames, segment
from light_synth import SynthParams, render
from morse_table import char_to_units


def sentence_trace(text, unit=0.3, fps=30.0, word_gap_after=()):
    """文字の間を3単位(語の間は7単位)にして文章の明るさの列を作る。"""
    t, intervals = 0.5, []
    for k, ch in enumerate(text):
        for j, code in enumerate(char_to_units(ch)):
            if j > 0:
                t += 3 * unit
            for m, e in enumerate(code):
                if m > 0:
                    t += unit
                length = unit * (3 if e == "-" else 1)
                intervals.append((t, t + length))
                t += length
        t += (7 if k in word_gap_after else 3) * unit
    p = SynthParams(unit=unit, fps=fps, exposure=0.6, phase=0.2, off_level=13.5, on_level=244.0)
    return render(intervals, t + 0.5, p)[1]


def test_unit_is_estimated_from_whole_signal():
    v = sentence_trace("ヨレコイロハ", unit=0.3)
    assert abs(estimate_unit_frames(binarize(v)) - 9) <= 1


def test_segments_split_at_char_gaps_and_mark_word_gaps():
    v = sentence_trace("イロハ", word_gap_after=(0,))
    segs = segment(binarize(v), 9.0)
    assert len(segs) == 3
    assert segs[1].gap_before >= 5 and 2 <= segs[2].gap_before < 5


def test_dakuten_is_its_own_segment():
    v = sentence_trace("ガ")  # カ + 濁点(間は3単位)
    assert len(segment(binarize(v), 9.0)) == 2


def test_single_frame_spikes_are_removed():
    v = sentence_trace("イ")
    v[3] = 240  # 雨粒で1コマだけ光る
    assert len(segment(binarize(v), 9.0)) == 1
