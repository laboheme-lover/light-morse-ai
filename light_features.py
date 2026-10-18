"""明るさの列を、モデルに入れる形(長さ一定の列)にする。学習と推論で必ずこの関数を共通に使う。

音のモールス(morse-ai)の morse_dataset.waveform_to_mel にあたる部分。学習と推論で入力の作り方が
少しでも違うとモデルの精度が落ちる(音で経験済み)ので、作り方はこのファイルの prepare_input だけにまとめる。

作り方
  (1) 明るさを 0〜1 にそろえる: 下位5%を0、上位95%を1とする(夕方・霧で差が縮んでいても同じ幅になる)
  (2) 信号の部分を切り出す: 中間(0.5)を超えた最初と最後のコマを探し、前後に1単位ずつ余白を付ける
  (3) 時間を「短点の長さ」でそろえる: 1単位(短点の長さ)が SAMPLES_PER_UNIT 点になるよう間引き・補間する。
      送る速さが違っても、同じ文字は同じ形になる。短点の長さは、1文字だけでは分からないので
      (長点しか無い「ヨ」と短点しか無い「ヌ」は区別できない)、音声全体・動画全体から測った値を外から渡す
  (4) 長さを INPUT_UNITS 単位(固定)にそろえる: 足りなければ0(消灯)で埋め、長ければ後ろを切る
"""

from __future__ import annotations

import numpy as np

SAMPLES_PER_UNIT = 6  # 1単位(短点の長さ)あたりの点の数
INPUT_UNITS = 24  # モデルに入れる長さ[単位]。最も長い文字「0」(-----)は19単位、前後の余白1単位ずつで21単位
INPUT_LENGTH = SAMPLES_PER_UNIT * INPUT_UNITS  # 144点


def normalize(values: np.ndarray) -> np.ndarray:
    """明るさを、下位5%=0・上位95%=1 にそろえる(範囲外は0〜1に収める)。"""
    v = np.asarray(values, dtype=np.float64)
    lo, hi = np.percentile(v, 5), np.percentile(v, 95)
    if hi - lo < 1e-6:
        return np.zeros_like(v)
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)


def prepare_input(values: np.ndarray, unit_frames: float) -> np.ndarray:
    """1文字分の明るさの列を、長さ INPUT_LENGTH の列(float32)にする。unit_frames = 短点の長さ[コマ]。"""
    x = normalize(values)
    on = np.where(x > 0.5)[0]
    if on.size:
        pad = int(round(unit_frames))
        x = x[max(0, on[0] - pad): on[-1] + pad + 1]
    # 1単位が SAMPLES_PER_UNIT 点になるように、時間方向に線形補間で並べ直す
    n_out = max(1, int(round(len(x) / unit_frames * SAMPLES_PER_UNIT)))
    src = np.linspace(0, len(x) - 1, n_out)
    x = np.interp(src, np.arange(len(x)), x)
    out = np.zeros(INPUT_LENGTH, dtype=np.float32)
    out[: min(INPUT_LENGTH, len(x))] = x[:INPUT_LENGTH]
    return out
