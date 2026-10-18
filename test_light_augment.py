import cv2
import numpy as np
import pytest

from light_augment import EFFECTS, SceneAugmenter
from light_roi import detect_signal_roi, extract_brightness
from test_light_roi import FPS, H, LAMP, W, morse_on_off


def write_augmented(path, effects, seed=3, seconds=6.0):
    """暗い背景に点滅するランプだけを置いた動画に、効果をかけて保存する。"""
    n = int(seconds * FPS)
    lamp = morse_on_off("-.-..", n)
    aug = SceneAugmenter(effects=effects, shape=(H, W), fps=FPS, seed=seed, avoid=(LAMP[0] - 10, LAMP[1] - 10, 20, 20))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(n):
        img = np.full((H, W, 3), 20, np.uint8)
        if lamp[i]:
            cv2.circle(img, LAMP, 9, (255, 255, 255), -1)
        writer.write(aug.apply(img, i))
    writer.release()
    return lamp


@pytest.mark.parametrize("effects", [(e,) for e in EFFECTS] + [("dusk", "rain", "lights"), ("fog", "lights", "noise")])
def test_lamp_is_still_found_under_effects(tmp_path, effects):
    path = tmp_path / "aug.mp4"
    lamp = write_augmented(path, effects)
    roi = detect_signal_roi(str(path))
    assert roi.x - 5 <= LAMP[0] <= roi.x + roi.w + 5 and roi.y - 5 <= LAMP[1] <= roi.y + roi.h + 5, (effects, roi)
    _, values = extract_brightness(str(path), roi)
    threshold = (np.percentile(values, 95) + np.percentile(values, 5)) / 2
    assert ((values > threshold) == (lamp > 0)).mean() > 0.9, effects


def test_effects_reduce_contrast_as_intended():
    frame = np.full((H, W, 3), 20, np.uint8)
    cv2.circle(frame, LAMP, 9, (255, 255, 255), -1)
    base = float(frame[LAMP[1], LAMP[0], 0]) - 20
    for effect in ("dusk", "fog"):
        out = SceneAugmenter(effects=(effect,), shape=(H, W), seed=1).apply(frame, 0).astype(float)
        diff = out[LAMP[1], LAMP[0]].mean() - out[10, 10].mean()
        assert 0 < diff < base, effect  # ランプはまだ明るいが、背景との差は小さくなる


def test_same_seed_gives_same_frames():
    frame = np.zeros((H, W, 3), np.uint8)
    a = SceneAugmenter(effects=("rain", "lights"), shape=(H, W), seed=5).apply(frame, 7)
    b = SceneAugmenter(effects=("rain", "lights"), shape=(H, W), seed=5).apply(frame, 7)
    assert np.array_equal(a, b)


def test_unknown_effect_is_rejected():
    with pytest.raises(ValueError):
        SceneAugmenter(effects=("snow",), shape=(H, W))
