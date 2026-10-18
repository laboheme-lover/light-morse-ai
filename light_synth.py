"""光の点滅(モールス信号)を R10 で撮ったときの「ランプの明るさの波形」(1コマごとの明るさ)を合成する。

音のモールス(morse-ai)の generate_morse_audio.py にあたる部分。光では、点滅する光源を見つけたあとは
1コマごとの明るさの列になるので、学習データはこの明るさの列として作る(動画そのものを作るより桁違いに速い)。

カメラが写す仕組みの再現(実際の動画65本の測定に合わせた)
  - 1コマの明るさは、そのコマの露光時間のあいだにランプが点いていた割合で決まる
    (点灯の始めのコマが途中の明るさ 70〜90% になるのを再現する)
  - ランプは非常に明るく、点灯中は明るさが上限付近で振り切れる(実測: 消灯 約13.5、点灯 約244)
  - コマの間隔は 30.3コマ/秒前後で、わずかに揺れる

悪条件(雑音)の再現(add_conditions)
  dusk    明け方・夕方: 背景(消灯時)が明るくなり、点灯との差が縮む。暗所のざらつきが増える
  fog     霧: 差が縮み、光がにじんで点灯・消灯の境目がなまる
  rain    雨: 雨粒がランプを横切ったときの一瞬の暗くなり・光り
  lights  背景の光の混入: ちらつく街灯(数Hzの明滅)、横切る車のライト(数コマの明るい山)
  camera  カメラ: 露出の自動調整による明るさのゆっくりした変化、手ぶれによるランプの一時的な外れ、コマ落ち
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from light_table import units_for_char

FPS = 30.3  # R10 の実測(30.31コマ/秒)
UNIT_RANGE = (0.15, 0.6)  # 短点の長さ[秒]。実測は 0.3 秒。これを中心に、ほかの速さにも対応できるよう幅を持たせる
TIMING_JITTER = 0.10  # 短点・長点・間の長さの揺らぎ(±10%。音の学習データと同じ)
OFF_LEVEL = 13.5  # 消灯時の明るさ(実測)
ON_LEVEL = 244.0  # 点灯時の明るさ(実測)
LEAD_SECONDS = (0.3, 1.0)  # 信号の前後の消灯の長さ[秒]
CONDITIONS = ("dusk", "fog", "rain", "lights", "camera")


@dataclass
class SynthParams:
    """1件の合成の条件(1文字の中では一定)。"""

    unit: float  # 短点の長さ[秒]
    fps: float
    exposure: float  # 露光時間(1コマの間隔に対する割合)
    phase: float  # 最初のコマの時刻のずれ(0〜1コマ)
    off_level: float
    on_level: float
    rise: float = 0.02  # ランプが点くときに明るくなるまでの時定数[秒]
    fall: float = 0.003  # 消えるときの時定数[秒]


# カメラの明るさの写り方(少し点いただけでも明るく写る振り切れ)の強さ。
# ランプの点き方・消え方の時定数(RISE_RANGE / FALL_RANGE)と合わせて、実際の動画65本の
# 「点灯の始めのコマの明るさ」(中央値 0.90)と「消灯の始めのコマ」(平均 0.04)に合うよう選んだ。
RESPONSE_POWER = 2.5
RISE_RANGE = (0.01, 0.04)
FALL_RANGE = (0.001, 0.005)
EXPOSURE_RANGE = (0.3, 1.0)
_STEP = 0.001  # ランプの明るさを計算する時間の刻み[秒]


def on_intervals(units: list[str], unit: float, rng: np.random.Generator, jitter: float = TIMING_JITTER,
                 start: float = 0.0) -> tuple[list[tuple[float, float]], float]:
    """符号の列(1文字なら1つ、濁音なら2つ)を、点灯している時間の区間 [(始め, 終わり), ...] にする。

    短点1単位・長点3単位・符号の中の間1単位・符号と符号の間(濁音の基本仮名と濁点の間)3単位。
    戻り値は (区間の列, 最後の点灯が終わった時刻)。
    """
    def jit(x):
        return x * (1 + rng.uniform(-jitter, jitter))

    t = start
    intervals = []
    for k, code in enumerate(units):
        if k > 0:
            t += jit(3 * unit)
        for j, element in enumerate(code):
            if j > 0:
                t += jit(unit)
            length = jit(unit * (3 if element == "-" else 1))
            intervals.append((t, t + length))
            t += length
    return intervals, t


def render(intervals: list[tuple[float, float]], duration: float, p: SynthParams) -> tuple[np.ndarray, np.ndarray]:
    """点灯の区間を、カメラで撮ったときの1コマごとの明るさにする。戻り値は (各コマの時刻, 明るさ)。"""
    # (1) ランプの明るさ(0〜1)を1msごとに求める。点灯の区間では時定数 rise で1に近づき、
    #     消灯の区間では時定数 fall で0に近づく(区間ごとに指数関数で計算する)
    tt = np.arange(0.0, duration, _STEP)
    lamp = np.zeros_like(tt)
    level = 0.0
    edges = sorted({0.0, duration, *[x for ab in intervals for x in ab]})
    on_at = lambda t: any(a <= t < b for a, b in intervals)  # noqa: E731
    for s, e in zip(edges[:-1], edges[1:]):
        idx = (tt >= s) & (tt < e)
        dt = tt[idx] - s
        if on_at(s):
            lamp[idx] = 1 - (1 - level) * np.exp(-dt / p.rise)
            level = 1 - (1 - level) * np.exp(-(e - s) / p.rise)
        else:
            lamp[idx] = level * np.exp(-dt / p.fall)
            level = level * np.exp(-(e - s) / p.fall)
    # (2) 各コマの露光時間のあいだのランプの明るさの平均(累積和の差で求める)
    n = int(duration * p.fps)
    frame_end = (np.arange(n) + p.phase) / p.fps  # 各コマの露光が終わる時刻
    exposure = p.exposure / p.fps
    cumulative = np.r_[0.0, np.cumsum(lamp)] * _STEP
    i_end = np.clip(np.round(frame_end / _STEP).astype(int), 0, len(lamp))
    i_start = np.clip(np.round((frame_end - exposure) / _STEP).astype(int), 0, len(lamp))
    fraction = np.clip((cumulative[i_end] - cumulative[i_start]) / exposure, 0, 1)
    # (3) ランプは明るく、少し点いていただけでも明るく写る(振り切れ)
    response = 1 - (1 - fraction) ** RESPONSE_POWER
    return frame_end, p.off_level + (p.on_level - p.off_level) * response


def sample_params(rng: np.random.Generator, unit: float | None = None) -> SynthParams:
    return SynthParams(
        unit=float(unit if unit is not None else rng.uniform(*UNIT_RANGE)),
        fps=float(rng.uniform(29.8, 30.4)),
        exposure=float(rng.uniform(*EXPOSURE_RANGE)),
        phase=float(rng.uniform(0, 1)),
        off_level=float(OFF_LEVEL + rng.normal(0, 2)),
        on_level=float(min(255.0, ON_LEVEL + rng.normal(0, 5))),
        rise=float(rng.uniform(*RISE_RANGE)),
        fall=float(rng.uniform(*FALL_RANGE)),
    )


def synthesize_char(char: str, rng: np.random.Generator, params: SynthParams | None = None) -> tuple[np.ndarray, SynthParams]:
    """1文字分の明るさの列を合成する(前後に消灯の余白を付ける)。"""
    p = params or sample_params(rng)
    lead = rng.uniform(*LEAD_SECONDS)
    intervals, end = on_intervals(units_for_char(char), p.unit, rng, start=lead)
    duration = end + rng.uniform(*LEAD_SECONDS)
    _, values = render(intervals, duration, p)
    return values, p


def add_conditions(values: np.ndarray, conditions: tuple[str, ...], rng: np.random.Generator, fps: float = FPS) -> np.ndarray:
    """明るさの列に悪条件をかける。強さは毎回抽選する。"""
    v = values.astype(np.float64).copy()
    n = len(v)
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    if "fog" in conditions:
        # 霧: 光がにじみ(前後のコマに広がる)、もやで差が縮む
        k = int(rng.integers(1, 4))
        kernel = np.exp(-np.abs(np.arange(-k, k + 1)) / max(k / 2, 0.5))
        v = np.convolve(v, kernel / kernel.sum(), mode="same")
        density = rng.uniform(0.3, 0.8)
        v = (1 - density) * v + density * rng.uniform(150, 210)
    if "dusk" in conditions:
        # 夕方・明け方: 背景が明るくなり、点灯との差が縮む(差が元の 10〜50% まで)
        gain = rng.uniform(0.1, 0.5)
        v = lo + (v - lo) * gain + rng.uniform(40, 160)
    if "lights" in conditions:
        t = np.arange(n) / fps
        if rng.random() < 0.8:  # ちらつく街灯の光が混ざる(数Hzの明滅)
            v += rng.uniform(5, 40) * np.sin(2 * np.pi * rng.uniform(0.5, 6) * t + rng.uniform(0, 6.3))
        for _ in range(int(rng.integers(0, 3))):  # 車のライトが横切る(数コマの明るい山)
            c, w = rng.uniform(0, n), rng.uniform(2, 10)
            v += rng.uniform(30, 150) * np.exp(-0.5 * ((np.arange(n) - c) / w) ** 2)
    if "rain" in conditions:
        # 雨粒が横切る: 一瞬(1〜2コマ)暗くなる・光る
        for i in rng.choice(n, size=int(n * rng.uniform(0.02, 0.1)), replace=False):
            v[i: i + int(rng.integers(1, 3))] += rng.uniform(-0.6, 0.3) * (hi - lo)
    if "camera" in conditions:
        # 露出の自動調整: 明るさ全体がゆっくり上下する
        v *= 1 + rng.uniform(0.05, 0.25) * np.sin(2 * np.pi * rng.uniform(0.05, 0.5) * np.arange(n) / fps + rng.uniform(0, 6.3))
        # 手ぶれ: ランプが一時的に検知した領域から外れて暗くなる
        if rng.random() < 0.5:
            s = int(rng.integers(0, n))
            v[s: s + int(rng.integers(2, 8))] *= rng.uniform(0.3, 0.8)
        # コマ落ち: 1コマ抜けて、前のコマが2回続く
        for i in rng.choice(n, size=int(rng.integers(0, max(1, n // 60) + 1)), replace=False):
            if i > 0:
                v[i] = v[i - 1]
    # センサーの雑音(暗いほど目立つ)。条件が悪いほど強くする
    sigma = rng.uniform(0.5, 2.0) + (rng.uniform(3, 10) if {"dusk", "fog"} & set(conditions) else 0)
    v += rng.normal(0, sigma, n)
    return np.clip(v, 0, 255)
