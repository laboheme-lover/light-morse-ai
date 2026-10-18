import numpy as np

from app import best_preview_frame, build_char_table, draw_candidates, render_text_html, zoom_crop
from light_roi import Roi, rank_signal_rois
from light_sequence import Segment, SequenceResult
from test_light_roi import LAMP, make_video


def test_preview_frame_shows_lamp_on(tmp_path):
    path = tmp_path / "scene.mp4"
    make_video(path)
    rois = rank_signal_rois(str(path))
    frame = best_preview_frame(str(path), rois[0])
    assert frame[LAMP[1], LAMP[0]].mean() > 200  # 点灯中のコマ
    img = draw_candidates(frame, rois, selected=0)
    assert img.shape == frame.shape
    assert zoom_crop(frame, rois[0]).shape[0] > rois[0].h


def _result():
    segs = [Segment(10, 20, float("inf"), "カ", 0.95), Segment(40, 45, 3.0, "゛", 0.3), Segment(70, 80, 3.0, "ア", 0.9)]
    return SequenceResult(text="ガア", segments=segs, unit_frames=9.0)


def test_char_table_marks_low_confidence():
    table = build_char_table(_result(), fps=30.0, low_confidence=0.5)
    assert table["文字"].tolist() == ["カ", "゛", "ア"]
    assert table["要確認"].tolist() == ["", "要確認", ""]
    assert table["前の消灯[単位]"].isna().tolist() == [True, False, False]


def test_text_html_highlights_low_confidence():
    out = render_text_html(_result(), 0.5)
    assert "ガ" in out and "゛" not in out  # 濁音は1文字にまとめる
    assert 'title="確信度 0.30">ガ<' in out  # 確信度は2区間のうち低いほう
