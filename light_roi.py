"""動画の中から、モールス信号で点滅している光源を自動で見つけ(フォーカス)、その明るさの時間変化を取り出す。

処理の流れ
  (1) read_gray_frames: 動画を縮小・白黒にして読み込む(検知は縮小した画像で十分で、速い)
  (2) find_candidates: ピクセルごとに「明るさがどれだけ揺れたか」(時間方向の標準偏差)を求め、
      揺れの大きい場所を光源の候補にする。止まっている光(街灯・窓明かり)は揺れが小さいので候補にならない
  (3) score_candidate: 候補ごとに明るさの波形を取り出し、「点灯と消灯の2つの明るさにはっきり分かれるか」
      (モールス信号はオン/オフの2値)で点数を付ける。ちらつく街灯・横切る車のライトなどは2値になりにくい
  (4) detect_signal_roi: 点数が最も高い候補を、点滅する光源の領域(ROI)として返す
  (5) extract_brightness: 元の大きさの動画で、ROIの中の明るさを1コマごとに取り出す(時刻つき)

音のモールス(morse-ai)では「音量の変化」から文字を区切った。光では、この明るさの変化が同じ役割をする。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DETECT_WIDTH = 320  # 検知に使う縮小後の幅[ピクセル](1920×1080 → 320×180)
MAX_DETECT_FRAMES = 900  # 検知に使う最大コマ数(30コマ/秒で約30秒)。長い動画は等間隔に間引く
TOP_CANDIDATES = 5  # 点数を付ける候補の数
MIN_ROI_SIZE = 8  # 元の動画でのROIの最小の大きさ[ピクセル]
MAX_ASPECT = 4.0  # 候補の縦横比の上限(これより細長いものは動く光の通り道とみなす)
MAX_SIZE_RATIO = 0.25  # 候補の大きさの上限(画面の幅・高さに対する割合)
CANDIDATE_SEARCHES = 15  # 揺れの大きい場所を探す回数(除いた候補の分も含む)
DETECT_SECONDS = 60.0  # 光源を探すのに使う、動画の最初の秒数
PERIODIC_MIN_CYCLES = 6  # 規則正しい点滅とみなす最少の繰り返し回数
PERIODIC_MAX_CV = 0.2  # 点灯・消灯の長さのばらつきがこれ未満なら「規則正しい」
PERIODIC_PENALTY = 0.2  # 規則正しく点滅し続ける光の点数に掛ける値


@dataclass
class Roi:
    """元の動画の座標での領域 (x, y, 幅, 高さ) と、検知の点数。"""

    x: int
    y: int
    w: int
    h: int
    score: float = 0.0


def read_gray_frames(
    path: str, width: int = DETECT_WIDTH, max_frames: int = MAX_DETECT_FRAMES, max_seconds: float = DETECT_SECONDS
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """動画を縮小した白黒のコマの配列 (コマ数, 高さ, 幅) にする。戻り値は (コマの配列, コマ/秒, 元の(幅, 高さ))。

    長い動画(文章を送った数分〜数十分の動画)は、最初の max_seconds 秒だけで光源を探す。
    ランプとカメラの位置は録画中に変わらない前提(三脚で固定)。
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise OSError(f"動画を開けません: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    total = min(total, int(max_seconds * fps)) if total else int(max_seconds * fps)
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    height = max(1, round(src_h * width / src_w))
    # 長い動画は等間隔に間引いて読む(すべて読むと遅い)。点滅の検知には揺れの大きさが分かれば十分
    step = max(1, total // max_frames) if total else 1
    frames = []
    index = 0
    while index < total:
        ok = cap.grab()
        if not ok:
            break
        if index % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        index += 1
    cap.release()
    if not frames:
        raise ValueError(f"コマを読めませんでした: {path}")
    return np.stack(frames).astype(np.float32), fps, (src_w, src_h)


def find_candidates(frames: np.ndarray, top: int = TOP_CANDIDATES) -> list[tuple[int, int, int, int]]:
    """明るさの揺れ(時間方向の標準偏差)が大きい場所を、縮小画像の座標の (x, y, 幅, 高さ) で返す。"""
    std = frames.std(axis=0)
    std = cv2.GaussianBlur(std, (0, 0), 1.0)  # 1ピクセルだけの雑音で候補が決まらないよう少しぼかす
    candidates = []
    work = std.copy()
    for _ in range(CANDIDATE_SEARCHES):
        if len(candidates) >= top:
            break
        _, peak, _, (px, py) = cv2.minMaxLoc(work)
        if peak <= 1e-6:
            break
        # 揺れが最大値の半分以上ある、つながった領域を1つの光源とする
        mask = (work >= peak * 0.5).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        x, y, w, h, _ = stats[labels[py, px]]
        # 動く光(横切る車のライトなど)は、通り道が細長い筋になる。点滅するランプは小さくまとまって写るので、
        # 細長い領域や、画面の大きな部分を占める領域は候補にしない
        elongated = max(w, h) > MAX_ASPECT * min(w, h)
        too_large = w > frames.shape[2] * MAX_SIZE_RATIO or h > frames.shape[1] * MAX_SIZE_RATIO
        if not (elongated or too_large):
            candidates.append((int(x), int(y), int(w), int(h)))
        # 見つけた領域(と周り)を消して、次に揺れの大きい場所を探す
        pad = 3
        work[max(0, y - pad): y + h + pad, max(0, x - pad): x + w + pad] = 0
    return candidates


def two_level_score(trace: np.ndarray) -> float:
    """明るさの波形が「点灯/消灯の2つの明るさ」にどれだけはっきり分かれるか(0〜1)。

    大津の方法で2つに分けたとき、2つの明るさの差がばらつき全体のどれだけを説明するか(クラス間分散の割合)。
    オン/オフの信号はほぼ1、ゆっくり変わる光やランダムなちらつきは小さくなる。
    """
    t = np.asarray(trace, dtype=np.float64)
    if t.size < 4 or t.std() < 1e-6:
        return 0.0
    order = np.sort(t)
    total_var = t.var()
    best = 0.0
    # 並べた値のすべての区切り位置で、2つのグループのクラス間分散を調べる
    csum = np.cumsum(order)
    n = order.size
    for k in range(1, n):
        w0, w1 = k / n, (n - k) / n
        m0 = csum[k - 1] / k
        m1 = (csum[-1] - csum[k - 1]) / (n - k)
        best = max(best, w0 * w1 * (m0 - m1) ** 2)
    return float(best / total_var)


def periodic_blinker(trace: np.ndarray, min_cycles: int = PERIODIC_MIN_CYCLES) -> bool:
    """同じ長さの点灯と消灯を規則正しく繰り返し続ける光(ハザードランプ・警告灯など)か。

    モールス信号には長点(3単位)や文字の間(3単位)があり、点灯・消灯の長さがそろわない。
    1文字の中で点灯が6回以上あるのは「、」(.-.-.-)だけで、長さがそろわないので当てはまらない。
    ただし「ヌ」(....)のように同じ点灯が4回程度続くだけの短い信号は、タイミングだけでは区別できない。
    """
    t = np.asarray(trace, dtype=np.float64)
    if t.std() < 1e-6:
        return False
    on = t > (t.min() + t.max()) / 2
    lengths: list[tuple[bool, int]] = []
    start = 0
    for i in range(1, len(on) + 1):
        if i == len(on) or on[i] != on[start]:
            lengths.append((bool(on[start]), i - start))
            start = i
    inner = lengths[1:-1]  # 両端は動画の切れ目で長さが分からないので除く
    on_runs = np.array([n for is_on, n in inner if is_on], dtype=float)
    off_runs = np.array([n for is_on, n in inner if not is_on], dtype=float)
    if len(on_runs) < min_cycles or len(off_runs) < min_cycles - 1:
        return False
    cv = lambda a: a.std() / a.mean()  # noqa: E731  ばらつき(標準偏差 / 平均)
    return cv(on_runs) < PERIODIC_MAX_CV and cv(off_runs) < PERIODIC_MAX_CV


def score_candidate(frames: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """候補の領域の明るさの波形に、点滅の信号らしさの点数を付ける(2値に分かれる度合い × 揺れの大きさ)。

    規則正しく点滅し続ける光は、モールス信号ではないとみなして点数を下げる。
    """
    x, y, w, h = box
    trace = frames[:, y: y + h, x: x + w].reshape(len(frames), -1).max(axis=1)
    score = two_level_score(trace) * float(trace.std())
    if periodic_blinker(trace):
        score *= PERIODIC_PENALTY
    return score


def rank_signal_rois(path: str) -> list[Roi]:
    """点滅している光源の候補を、信号らしさの点数の高い順に返す(元の動画の座標)。

    アプリでは1番目を自動で選び、利用者が画面で別の候補に選び直せるようにする(フォーカス機能)。
    """
    frames, _fps, (src_w, src_h) = read_gray_frames(path)
    scale = src_w / frames.shape[2]
    candidates = find_candidates(frames)
    scored = sorted(((score_candidate(frames, c), c) for c in candidates), reverse=True)
    rois = []
    for score, (x, y, w, h) in scored:
        # 縮小画像の座標を元の動画の座標に戻し、光のにじみを含むよう周りに余白を付ける
        margin = 2
        x0 = max(0, int((x - margin) * scale))
        y0 = max(0, int((y - margin) * scale))
        x1 = min(src_w, int((x + w + margin) * scale))
        y1 = min(src_h, int((y + h + margin) * scale))
        rois.append(Roi(x0, y0, max(MIN_ROI_SIZE, x1 - x0), max(MIN_ROI_SIZE, y1 - y0), score))
    return rois


def detect_signal_roi(path: str) -> Roi:
    """動画の中で、モールス信号で点滅している光源の領域を見つける(元の動画の座標で返す)。"""
    rois = rank_signal_rois(path)
    if not rois:
        raise ValueError("点滅している光源が見つかりませんでした")
    return rois[0]


def extract_brightness(path: str, roi: Roi) -> tuple[np.ndarray, np.ndarray]:
    """ROIの中の明るさを1コマずつ取り出す。戻り値は (時刻[秒] の配列, 明るさ(0〜255) の配列)。

    明るさは、ROIの中で明るい上位10%のピクセルの平均。ランプが小さく写っていても、
    背景の暗いピクセルに薄められずに点灯・消灯の差が出る。
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise OSError(f"動画を開けません: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    times, values = [], []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        patch = cv2.cvtColor(frame[roi.y: roi.y + roi.h, roi.x: roi.x + roi.w], cv2.COLOR_BGR2GRAY).ravel()
        k = max(1, patch.size // 10)
        values.append(float(np.partition(patch, patch.size - k)[-k:].mean()))
        # 時刻はコマ番号から計算する(スマホの動画はコマの間隔がわずかに揺れるが、まずは一定とみなす)
        times.append(index / fps)
        index += 1
    cap.release()
    return np.asarray(times), np.asarray(values)
