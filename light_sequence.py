"""文章を送った長い動画の明るさの列を、1文字ずつに区切って読む。音の predict_sequence.py にあたる。

処理の流れ
  (1) 明るさを 0〜1 にそろえ、点灯/消灯をヒステリシス(上がるときは0.6、下がるときは0.4)で判定する。
      1コマだけの点灯・消灯(雨粒などの雑音)は前後に合わせて消す
  (2) 短点の長さを動画全体から測る: 点灯の長さを集め、短いほうから15%の位置の値とする
      (音の estimate_dot_seconds と同じ考え方。1文字だけでは分からないので全体から測る)
  (3) 消灯が 2単位以上続いたところで文字を区切る(符号の中の間は1単位、文字の間は3単位)。
      5単位以上なら語の区切り(7単位)とみなす
  (4) 区切った1文字ずつを light_features.prepare_input で入力の形にし、モデルで判定する
  (5) 「カ」+「゛」を「ガ」のように結合する(音と同じ morse_table.compose_dakuten)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from light_features import normalize, prepare_input
from morse_table import compose_dakuten

HIGH, LOW = 0.6, 0.4  # 点灯/消灯の判定のヒステリシス
DOT_PERCENTILE = 15
CHAR_GAP_UNITS = 2.0  # これ以上の消灯で文字を区切る
WORD_GAP_UNITS = 5.0  # これ以上の消灯は語の区切り


@dataclass
class Segment:
    start: int  # 最初の点灯のコマ
    end: int  # 最後の点灯のコマ(含む)
    gap_before: float  # 直前の消灯の長さ[単位]
    label: str = ""
    confidence: float = 0.0


@dataclass
class SequenceResult:
    text: str
    segments: list[Segment]
    unit_frames: float


def binarize(values: np.ndarray) -> np.ndarray:
    """明るさの列を点灯(True)/消灯(False)にする。1コマだけの点灯・消灯は消す。"""
    x = normalize(values)
    on = np.zeros(len(x), dtype=bool)
    state = False
    for i, v in enumerate(x):
        state = v > LOW if state else v > HIGH
        on[i] = state
    for i in range(1, len(on) - 1):
        if on[i] != on[i - 1] and on[i] != on[i + 1]:
            on[i] = on[i - 1]
    return on


def runs(on: np.ndarray) -> list[tuple[bool, int, int]]:
    """(点灯か, 始まりのコマ, 長さ) の並び。"""
    out, start = [], 0
    for i in range(1, len(on) + 1):
        if i == len(on) or on[i] != on[start]:
            out.append((bool(on[start]), start, i - start))
            start = i
    return out


def estimate_unit_frames(on: np.ndarray) -> float:
    lengths = [n for is_on, _, n in runs(on) if is_on]
    return float(np.percentile(lengths, DOT_PERCENTILE)) if lengths else 9.0


def segment(on: np.ndarray, unit_frames: float) -> list[Segment]:
    segments: list[Segment] = []
    gap = float("inf")
    for is_on, start, n in runs(on):
        if not is_on:
            gap = n / unit_frames
            continue
        if segments and gap < CHAR_GAP_UNITS:
            segments[-1].end = start + n - 1  # 同じ文字の続き
        else:
            segments.append(Segment(start, start + n - 1, gap))
    return segments


def recognize(values: np.ndarray, model, labels: list[str], unit_frames: float | None = None) -> SequenceResult:
    on = binarize(values)
    unit = unit_frames or estimate_unit_frames(on)
    segs = segment(on, unit)
    pad = int(round(2 * unit))
    xs = [prepare_input(values[max(0, s.start - pad): s.end + pad + 1], unit) for s in segs]
    if xs:
        with torch.no_grad():
            prob = torch.softmax(model(torch.from_numpy(np.stack(xs))[:, None, :]), dim=1).numpy()
        for s, p in zip(segs, prob):
            k = int(p.argmax())
            s.label, s.confidence = labels[k], float(p[k])
    text = compose_dakuten("".join(s.label for s in segs))
    return SequenceResult(text=text, segments=segs, unit_frames=unit)
