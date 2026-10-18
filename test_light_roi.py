import cv2
import numpy as np

from light_roi import detect_signal_roi, extract_brightness, two_level_score

FPS = 30
W, H = 640, 360
UNIT = 0.3  # 短点 300ms(実際のテスト動画と同じ)
LAMP = (400, 200)  # 点滅するランプの中心


def morse_on_off(code: str, n_frames: int, lead: float = 0.5) -> np.ndarray:
    """符号(例 '-.-..')をコマごとの点灯(1)/消灯(0)にする。短点1単位・長点3単位・間1単位。"""
    on = np.zeros(n_frames)
    t = lead
    for ch in code:
        if ch == " ":  # 空白は消灯1単位(3つ並べると文字の間の3単位に、前の間と合わせて4単位)
            t += UNIT
            continue
        length = UNIT * (3 if ch == "-" else 1)
        on[int(t * FPS): int((t + length) * FPS)] = 1
        t += length + UNIT
    return on


def make_video(path, seconds=6.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * FPS)
    lamp = morse_on_off("-.-..", n)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(n):
        img = np.full((H, W, 3), 25, np.float32) + rng.normal(0, 3, (H, W, 3))
        cv2.circle(img, (100, 80), 10, (230, 230, 230), -1)  # 止まった明かり(窓・街灯)
        flicker = 140 + 50 * np.sin(2 * np.pi * 2.7 * i / FPS)  # ちらつく街灯(電源の周波数がコマ数と干渉した明滅)
        cv2.circle(img, (200, 280), 10, (flicker,) * 3, -1)
        cx = int(-50 + i * (W + 100) / n)  # 横切る車のライト
        cv2.circle(img, (cx, 320), 8, (255, 255, 255), -1)
        if lamp[i]:
            cv2.circle(img, LAMP, 9, (255, 255, 255), -1)
        writer.write(np.clip(img, 0, 255).astype(np.uint8))
    writer.release()
    return lamp


def test_two_level_score_prefers_on_off_signal():
    on_off = np.r_[np.zeros(30), np.ones(20), np.zeros(10), np.ones(40)] * 200
    sine = 100 + 100 * np.sin(np.linspace(0, 12, 100))
    assert two_level_score(on_off) > 0.99
    assert two_level_score(sine) < 0.9
    assert two_level_score(np.full(50, 7.0)) == 0.0


def test_detects_blinking_lamp_among_other_lights(tmp_path):
    path = tmp_path / "scene.mp4"
    make_video(path)
    roi = detect_signal_roi(str(path))
    assert roi.x <= LAMP[0] <= roi.x + roi.w
    assert roi.y <= LAMP[1] <= roi.y + roi.h
    assert roi.w <= 80 and roi.h <= 80  # ランプの周りだけに絞れている


def test_extracted_brightness_follows_on_off(tmp_path):
    path = tmp_path / "scene.mp4"
    lamp = make_video(path)
    roi = detect_signal_roi(str(path))
    times, values = extract_brightness(str(path), roi)
    assert len(values) == len(lamp)
    assert np.allclose(np.diff(times), 1 / FPS)
    threshold = (values.max() + values.min()) / 2
    agree = ((values > threshold) == (lamp > 0)).mean()
    assert agree > 0.97  # 圧縮の影響で境目の1コマがずれる程度


def test_periodic_blinker_is_recognized_but_morse_is_not():
    from light_roi import periodic_blinker

    blinker = np.tile(np.r_[np.ones(6), np.zeros(6)], 15) * 200
    assert periodic_blinker(blinker)
    sentence = morse_on_off("-.-.." + "   " + ".-.-.-" + "   " + "--.-", 400) * 200  # 文字の間に3単位の消灯
    assert not periodic_blinker(sentence)
    nu = morse_on_off("....", 120) * 200  # 「ヌ」は点灯4回だけなので規則正しい点滅とはみなさない
    assert not periodic_blinker(nu)


def test_morse_lamp_is_chosen_over_periodic_blinker(tmp_path):
    from light_roi import rank_signal_rois

    n = int(8 * FPS)
    lamp = morse_on_off("-.-..", n)
    blink = (np.arange(n) % 12) < 6  # 0.4秒周期で点滅し続ける警告灯(ランプと同じ明るさ)
    path = tmp_path / "decoy.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(n):
        img = np.full((H, W, 3), 20, np.uint8)
        if lamp[i]:
            cv2.circle(img, LAMP, 9, (255, 255, 255), -1)
        if blink[i]:
            cv2.circle(img, (150, 120), 9, (255, 255, 255), -1)
        writer.write(img)
    writer.release()
    rois = rank_signal_rois(str(path))
    best = rois[0]
    assert best.x <= LAMP[0] <= best.x + best.w and best.y <= LAMP[1] <= best.y + best.h
    assert len(rois) >= 2  # 警告灯も候補には残る(アプリで選び直せる)
