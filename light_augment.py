"""実際の動画のコマに、悪条件の効果(雨・霧・明け方/夕方・背景の複数の光源・センサーの雑音)を加える。

音のモールス(morse-ai)で、合成音声に白色雑音や環境音(ESC-50)を混ぜて学習・評価したのと同じ考え方を、
映像に当てはめる。主な用途は、点滅する光源の自動検知(light_roi.py)が悪条件でも正しいランプを
見つけられるかを確かめることと、悪条件での明るさの波形を作ること。

使い方
    aug = SceneAugmenter(effects=("dusk", "rain"), shape=(1080, 1920), fps=30.3, seed=1)
    for i, frame in enumerate(frames):
        out = aug.apply(frame, i)
1本の動画の中では効果の強さ(雨の量・霧の濃さなど)を一定にし、雨粒や車のライトの位置だけをコマごとに動かす。

効果の一覧(EFFECTS)
    dusk    明け方・夕方: 背景が明るくなり、ランプとの明るさの差(コントラスト)が小さくなる。色味がオレンジ寄りになり、
            暗所のセンサーの雑音が増える
    fog     霧: 全体に白いもやがかかって差が小さくなり、光がにじむ(ぼかし + 明るい部分の広がり)
    rain    雨: 斜めの雨の筋がコマごとに動き、ランプの手前を横切ると一瞬暗くなったり光ったりする
    lights  背景の複数の光源: 止まった明かり・ちらつく街灯(電源の周波数とコマ数の干渉による数Hzの明滅)・
            横切る車のライト・別のリズムで点滅する光(誤検知を誘うおとり)
    noise   センサーの雑音と動画の圧縮の劣化(JPEGで代用)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

EFFECTS = ("dusk", "fog", "rain", "lights", "noise")


@dataclass
class _Light:
    kind: str  # static / flicker / car / decoy
    x: float
    y: float
    radius: int
    level: float  # 明るさ(0〜255)
    freq: float = 0.0  # flicker の明滅の周波数[Hz]
    vx: float = 0.0  # car の速さ[ピクセル/コマ]
    pattern: np.ndarray | None = None  # decoy の点灯(1)/消灯(0)の列(コマ単位)
    color: tuple[int, int, int] = (255, 255, 255)


@dataclass
class SceneAugmenter:
    """1本の動画に加える効果。強さは seed から1回だけ抽選し、その動画の中では一定にする。"""

    effects: tuple[str, ...]
    shape: tuple[int, int]  # (高さ, 幅)
    fps: float = 30.0
    seed: int = 0
    avoid: tuple[int, int, int, int] | None = None  # 光源を置かない領域(本物のランプの ROI)
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        unknown = set(self.effects) - set(EFFECTS)
        if unknown:
            raise ValueError(f"未対応の効果: {sorted(unknown)}")
        rng = np.random.default_rng(self.seed)
        self._rng = rng
        h, w = self.shape
        p = self.params
        # 効果の強さを抽選する(1本の動画で一定)
        p.setdefault("dusk_level", float(rng.uniform(60, 140)))  # 背景の明るさの持ち上げ
        p.setdefault("dusk_gain", float(rng.uniform(0.5, 0.8)))  # ランプとの差を小さくする割合
        p.setdefault("fog_density", float(rng.uniform(0.3, 0.7)))  # もやの濃さ(0〜1)
        p.setdefault("rain_drops", int(rng.integers(150, 600)))  # 1コマあたりの雨の筋の数
        p.setdefault("noise_sigma", float(rng.uniform(4, 12)))  # センサーの雑音の強さ
        p.setdefault("jpeg_quality", int(rng.integers(30, 70)))
        self._lights = self._make_lights(rng, h, w) if "lights" in self.effects else []

    # ---- 背景の光源
    def _free_position(self, rng, h, w, margin=40):
        """本物のランプ(avoid)から離れた位置を選ぶ。"""
        for _ in range(100):
            x, y = rng.uniform(margin, w - margin), rng.uniform(margin, h - margin)
            if self.avoid is None:
                return x, y
            ax, ay, aw, ah = self.avoid
            if not (ax - 3 * aw < x < ax + 4 * aw and ay - 3 * ah < y < ay + 4 * ah):
                return x, y
        return float(margin), float(margin)

    def _make_lights(self, rng, h, w) -> list[_Light]:
        scale = w / 1920
        lights = []
        for _ in range(int(rng.integers(2, 6))):  # 止まった明かり(窓・街灯)
            x, y = self._free_position(rng, h, w)
            lights.append(_Light("static", x, y, int(rng.integers(6, 30) * scale) + 2, float(rng.uniform(150, 255)),
                                 color=(int(rng.uniform(150, 255)), int(rng.uniform(180, 255)), 255)))
        for _ in range(int(rng.integers(1, 3))):  # ちらつく街灯(蛍光灯・LEDの明滅が30コマ/秒で写ると数Hzに見える)
            x, y = self._free_position(rng, h, w)
            lights.append(_Light("flicker", x, y, int(rng.integers(8, 25) * scale) + 2, float(rng.uniform(120, 230)),
                                 freq=float(rng.uniform(0.5, 6.0))))
        if rng.random() < 0.7:  # 横切る車のライト
            y = rng.uniform(h * 0.6, h * 0.95)
            speed = rng.uniform(4, 20) * scale * rng.choice([-1, 1])
            lights.append(_Light("car", -50.0 if speed > 0 else w + 50.0, y, int(rng.integers(10, 25) * scale) + 2,
                                 255.0, vx=float(speed)))
        if rng.random() < 0.5:  # 別のリズムで点滅する光(おとり)。周期的で、モールスの長短の並びにはならない
            x, y = self._free_position(rng, h, w)
            period = int(rng.integers(8, 40))
            duty = rng.uniform(0.2, 0.8)
            pattern = (np.arange(10000) % period) < period * duty
            lights.append(_Light("decoy", x, y, int(rng.integers(6, 20) * scale) + 2, float(rng.uniform(180, 255)),
                                 pattern=pattern.astype(float)))
        return lights

    def _draw_lights(self, img: np.ndarray, i: int) -> None:
        overlay = np.zeros_like(img)
        for L in self._lights:
            level = L.level
            x = L.x
            if L.kind == "flicker":
                level = L.level * (0.55 + 0.45 * np.sin(2 * np.pi * L.freq * i / self.fps))
            elif L.kind == "car":
                x = L.x + L.vx * i
            elif L.kind == "decoy":
                level = L.level * L.pattern[i % len(L.pattern)]
            if level <= 0:
                continue
            color = tuple(float(c) * level / 255 for c in L.color)
            cv2.circle(overlay, (int(x), int(L.y)), L.radius, color, -1, lineType=cv2.LINE_AA)
        # 光のにじみ: ぼかした光を重ねて、光源の周りがぼんやり明るくなるようにする
        glow = cv2.GaussianBlur(overlay, (0, 0), 6)
        np.maximum(img, overlay, out=img)
        img += glow * 0.6

    # ---- 1コマに効果をかける
    def apply(self, frame: np.ndarray, i: int) -> np.ndarray:
        """BGR の1コマ(uint8)に効果をかけて返す。i はコマ番号(雨粒・車の位置・明滅の計算に使う)。"""
        p = self.params
        img = frame.astype(np.float32)
        if "lights" in self.effects:
            self._draw_lights(img, i)
        if "dusk" in self.effects:
            # 明け方・夕方: 空の明るさで背景全体が持ち上がり、ランプとの差が縮む。色はオレンジ寄り
            tint = np.array([0.75, 0.9, 1.1], np.float32)  # B, G, R
            img = img * p["dusk_gain"] + p["dusk_level"] * tint
        if "fog" in self.effects:
            # 霧: 白いもやを混ぜて差を小さくし、明るい部分をにじませる
            d = p["fog_density"]
            glow = cv2.GaussianBlur(img, (0, 0), 8 + 20 * d)
            img = (1 - d) * img + d * 200.0 + d * 0.5 * np.maximum(glow - 120, 0)
        if "rain" in self.effects:
            img = self._rain(img, i)
        if "noise" in self.effects:
            img = img + self._rng.normal(0, p["noise_sigma"], img.shape).astype(np.float32)
        out = np.clip(img, 0, 255).astype(np.uint8)
        if "noise" in self.effects:
            ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, p["jpeg_quality"]])
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return out

    def _rain(self, img: np.ndarray, i: int) -> np.ndarray:
        """斜めの雨の筋。コマごとに位置を変える(同じ i なら同じ雨)。"""
        h, w = img.shape[:2]
        rng = np.random.default_rng(self.seed * 100003 + i)
        streaks = np.zeros((h, w), np.float32)
        n = self.params["rain_drops"]
        xs, ys = rng.uniform(0, w, n), rng.uniform(0, h, n)
        length = rng.uniform(15, 45, n) * (w / 1920)
        for x, y, L in zip(xs, ys, length):
            cv2.line(streaks, (int(x), int(y)), (int(x + 0.25 * L), int(y + L)), float(rng.uniform(0.3, 1.0)), 1,
                     lineType=cv2.LINE_AA)
        streaks = cv2.GaussianBlur(streaks, (0, 0), 0.8)[..., None]
        # 雨の筋は周りの光を受けて白っぽく光り、手前の光をわずかに遮る
        return img * (1 - 0.35 * streaks) + 180.0 * 0.35 * streaks
