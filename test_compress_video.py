import cv2

from compress_video import compress
from light_roi import extract_brightness, rank_signal_rois
from test_light_roi import FPS, H, LAMP, W, make_video


def test_compress_keeps_frames_and_lamp(tmp_path):
    src = tmp_path / "scene.mp4"
    lamp = make_video(src)
    dst = tmp_path / "small" / "scene.mp4"
    r = compress(src, dst, width=W // 2, progress=lambda m: None)

    cap = cv2.VideoCapture(str(dst))
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == W // 2 and int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == H // 2
    assert abs(cap.get(cv2.CAP_PROP_FPS) - FPS) < 0.01  # 点滅の長さはコマ数で測るので、コマの速さは変えない
    cap.release()
    assert r["frames"] == len(lamp)
    assert not list(dst.parent.glob("*.part.mp4"))

    roi = rank_signal_rois(str(dst))[0]  # 縮めても同じランプを見つけ、点灯・消灯を読める
    assert roi.x <= LAMP[0] // 2 <= roi.x + roi.w and roi.y <= LAMP[1] // 2 <= roi.y + roi.h
    _, values = extract_brightness(str(dst), roi)
    lit = lamp.astype(bool)
    on, off = values[lit].mean(), values[~lit].mean()
    assert on - off > 100


def test_compress_does_not_enlarge(tmp_path):
    src = tmp_path / "scene.mp4"
    make_video(src, seconds=1.0)
    r = compress(src, tmp_path / "out.mp4", width=W * 2, progress=lambda m: None)
    assert r["size"] == (W, H)
