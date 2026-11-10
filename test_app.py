import io
import subprocess
from pathlib import Path

import numpy as np

from app import (best_preview_frame, build_char_table, draw_candidates, ensure_model, initial_dir_for,
                 pick_video_file, render_text_html, save_upload, zoom_crop)
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


def test_pick_video_file_returns_chosen_path_or_empty():
    calls = []

    def fake(stdout):
        def run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr=b"")
        return run

    chosen = pick_video_file("C:/videos", runner=fake("C:/動画/文章_sentence.mp4\n".encode("utf-8")))
    assert chosen == str(Path("C:/動画/文章_sentence.mp4"))  # Windows では区切り文字を \ に直す
    assert calls[0][-1] == "C:/videos"  # 最初に開く場所を渡す
    assert pick_video_file(runner=fake(b"")) == ""  # 取り消し


def test_initial_dir_for_uses_folder_of_current_video(tmp_path):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"")
    assert initial_dir_for(f'"{video}"', default=tmp_path / "none") == str(tmp_path)
    assert initial_dir_for("", default=tmp_path) == str(tmp_path)
    assert initial_dir_for("", default=tmp_path / "none") == ""


def test_ensure_model_downloads_once_and_cleans_up_on_failure(tmp_path):
    source = tmp_path / "release.pt"
    source.write_bytes(b"weights")
    dst = tmp_path / "models" / "best_model.pt"
    ensure_model(dst, url=source.as_uri())
    assert dst.read_bytes() == b"weights"
    ensure_model(dst, url=(tmp_path / "missing.pt").as_uri())  # あるときは取りに行かない
    other = tmp_path / "other" / "best_model.pt"
    try:
        ensure_model(other, url=(tmp_path / "missing.pt").as_uri())
        raise AssertionError("取ってこられないときは例外になるはず")
    except OSError:
        pass
    assert not other.exists() and not list(other.parent.glob("*.part"))  # 途中のファイルを残さない


class _Upload(io.BytesIO):
    """Streamlit の UploadedFile の代わり(必要な属性だけ)"""

    def __init__(self, data: bytes, name: str, file_id: str):
        super().__init__(data)
        self.name, self.file_id, self.size = name, file_id, len(data)


def test_save_upload_keeps_uploads_with_same_name_apart(tmp_path):
    a = save_upload(_Upload(b"aaaa", "video.MP4", "id1"), tmp_path)
    b = save_upload(_Upload(b"bbbb", "video.MP4", "id2"), tmp_path)  # 同じ名前・同じ大きさの別の動画
    assert a != b and a.suffix == ".mp4"
    assert a.read_bytes() == b"aaaa" and b.read_bytes() == b"bbbb"
