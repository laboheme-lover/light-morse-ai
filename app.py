"""光のモールス信号の読み取りアプリ(Streamlit)。音の morse-ai/app.py にあたる。

R10 で撮った動画を渡すと、
  (1) 点滅している光源の候補を自動で見つけ、映像の上に番号つきの枠で示す(1番を自動で選ぶ。選び直せる)
  (2) 選んだ光源の明るさを1コマずつ取り出す
  (3) 1文字ずつに区切ってモデルで読み、文章として表示する(確信度の低い文字は色を付ける)

起動: streamlit run app.py
文章の動画は数GBになるので、ファイルの場所(パス)を渡すのが基本(「参照…」で選ぶか、入力する)。1GBまでならアップロードもできる。
"""

from __future__ import annotations

import html
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from light_model import MorseLightNet
from light_roi import DETECT_SECONDS, Roi, extract_brightness, rank_signal_rois
from light_sequence import SequenceResult, recognize
from morse_table import compose_pair

matplotlib.use("Agg")
for _font in ("Yu Gothic", "Meiryo", "MS Gothic"):
    if _font in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = _font
        break

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
DEFAULT_LOW_CONFIDENCE = 0.5
ONLINE_ONLY = 0x400000 | 0x1000  # OneDrive の「オンラインのみ」のファイルの属性
MAX_CANDIDATES_SHOWN = 5
DEFAULT_VIDEO_DIR = Path.home() / "OneDrive" / "光点滅テスト"

# ファイルを選ぶ画面を開いて、選んだ場所を UTF-8 で書き出す(取り消したら何も書かない)
PICK_SCRIPT = r'''
import sys
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)  # ブラウザの後ろに隠れないよう、いちばん手前に出す
path = filedialog.askopenfilename(parent=root, title="動画を選ぶ", initialdir=sys.argv[1] or None,
                                  filetypes=[("動画", "*.mp4 *.mov"), ("すべてのファイル", "*.*")])
sys.stdout.buffer.write(path.encode("utf-8"))
'''


# --- 処理(画面から独立。単体テストできる) ------------------------------------


def initial_dir_for(current: str, default: Path = DEFAULT_VIDEO_DIR) -> str:
    """ファイルを選ぶ画面を最初に開く場所。入力済みの動画のフォルダ、なければ光点滅テストのフォルダ。"""
    folder = Path(current.strip().strip('"')).parent if current.strip() else None
    if folder is not None and folder.is_dir():
        return str(folder)
    return str(default) if default.is_dir() else ""


def pick_video_file(initial_dir: str = "", runner=subprocess.run) -> str:
    """Windows のファイルを選ぶ画面を開き、選んだ動画の場所を返す(取り消したら空文字)。
    アプリはこの PC の中で動いているので、画面もこの PC に出る。Streamlit のスレッドで Tk を動かさないよう別のプロセスで開く。"""
    done = runner([sys.executable, "-c", PICK_SCRIPT, initial_dir], capture_output=True, check=False)
    chosen = done.stdout.decode("utf-8").strip()
    return str(Path(chosen)) if chosen else ""


def ensure_local(path: str) -> None:
    """OneDrive の「オンラインのみ」の動画を、先に読み通してダウンロードを済ませる(動画の読み込みの時間切れを防ぐ)。"""
    if getattr(os.stat(path), "st_file_attributes", 0) & ONLINE_ONLY:
        with open(path, "rb") as f:
            while f.read(64 * 1024 * 1024):
                pass


def best_preview_frame(path: str, roi: Roi, seconds: float = DETECT_SECONDS, step: int = 5) -> np.ndarray:
    """最初の数十秒の中で、選んだ光源が最も明るく写っているコマ(点灯中)を返す(BGR)。"""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    best, best_value, i = None, -1.0, 0
    while i < seconds * fps:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                value = float(frame[roi.y: roi.y + roi.h, roi.x: roi.x + roi.w].max())
                if value > best_value:
                    best, best_value = frame, value
        i += 1
    cap.release()
    if best is None:
        raise ValueError("動画のコマを読めませんでした")
    return best


def draw_candidates(frame: np.ndarray, rois: list[Roi], selected: int) -> np.ndarray:
    """映像に候補の枠と番号を描く(選んだ候補は緑の太枠、ほかは黄色)。RGB で返す。"""
    img = frame.copy()
    thick = max(2, img.shape[1] // 400)
    for k, r in enumerate(rois):
        color = (0, 255, 0) if k == selected else (0, 220, 255)
        pad = 20
        cv2.rectangle(img, (r.x - pad, r.y - pad), (r.x + r.w + pad, r.y + r.h + pad), color, thick * (2 if k == selected else 1))
        cv2.putText(img, str(k + 1), (r.x - pad, r.y - pad - 10), cv2.FONT_HERSHEY_SIMPLEX, thick * 0.8, color, thick)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def zoom_crop(frame: np.ndarray, roi: Roi, margin: int = 80) -> np.ndarray:
    """選んだ光源の周りを切り出す(RGB)。"""
    y0, x0 = max(0, roi.y - margin), max(0, roi.x - margin)
    crop = frame[y0: roi.y + roi.h + margin, x0: roi.x + roi.w + margin]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def load_model(models_dir: Path = MODELS_DIR) -> tuple[MorseLightNet, list[str]]:
    mapping = json.loads((models_dir / "label_mapping.json").read_text(encoding="utf-8"))
    state = torch.load(models_dir / "best_model.pt", map_location="cpu", weights_only=False)
    model = MorseLightNet(num_classes=len(mapping["labels"]))
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, mapping["labels"]


def build_char_table(result: SequenceResult, fps: float, low_confidence: float) -> pd.DataFrame:
    """文字ごとの詳細(区間ごと。濁点は別の行)。"""
    rows = []
    for k, s in enumerate(result.segments, start=1):
        rows.append({
            "番号": k,
            "文字": s.label,
            "確信度": round(s.confidence, 3),
            "開始[秒]": round(s.start / fps, 2),
            "終了[秒]": round((s.end + 1) / fps, 2),
            "前の消灯[単位]": None if np.isinf(s.gap_before) else round(s.gap_before, 1),
            "要確認": "要確認" if s.confidence < low_confidence else "",
        })
    return pd.DataFrame(rows)


def render_text_html(result: SequenceResult, low_confidence: float) -> str:
    """読み取った文章を、確信度の低い区間に色を付けた HTML にする(濁点も1区間として扱う)。"""
    parts = []
    segs = result.segments
    i = 0
    while i < len(segs):
        # 「カ」+「゛」のような2区間は「ガ」の1文字にまとめる(確信度は低いほう)
        composed = compose_pair(segs[i].label, segs[i + 1].label) if i + 1 < len(segs) else None
        if composed:
            label, confidence, i = composed, min(segs[i].confidence, segs[i + 1].confidence), i + 2
        else:
            label, confidence, i = segs[i].label, segs[i].confidence, i + 1
        text = html.escape(label)
        if confidence < low_confidence:
            parts.append(f'<span style="background:rgba(255,170,0,0.35);border-bottom:3px solid #e67e00;" '
                         f'title="確信度 {confidence:.2f}">{text}</span>')
        else:
            parts.append(text)
    return ('<div style="font-size:1.9rem;line-height:2.4rem;letter-spacing:0.08em;word-break:break-all;">'
            + "".join(parts) + "</div>")


def plot_trace(values: np.ndarray, fps: float, result: SequenceResult, window: tuple[float, float] | None = None):
    """明るさの波形と、文字の区間(色の帯)を描く。"""
    fig, ax = plt.subplots(figsize=(11, 2.8))
    t = np.arange(len(values)) / fps
    ax.plot(t, values, lw=0.8)
    for k, s in enumerate(result.segments):
        a, b = s.start / fps, (s.end + 1) / fps
        if window and (b < window[0] or a > window[1]):
            continue
        ax.axvspan(a, b, color=("#c8e6c9" if k % 2 == 0 else "#bbdefb"), alpha=0.5)
        ax.text((a + b) / 2, values.max() * 1.02, s.label, ha="center", fontsize=9)
    if window:
        ax.set_xlim(*window)
    ax.set_xlabel("時刻[秒]")
    ax.set_ylabel("明るさ")
    fig.tight_layout()
    return fig


# --- キャッシュ -----------------------------------------------------------------


@st.cache_resource(show_spinner="モデルを読み込み中...")
def get_model(model_mtime: float):
    return load_model()


@st.cache_data(show_spinner="点滅している光源を探しています(最初の60秒)...")
def get_candidates(path: str, mtime: float):
    ensure_local(path)
    rois = rank_signal_rois(path)[:MAX_CANDIDATES_SHOWN]
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return rois, fps, frames


@st.cache_data(show_spinner="映像を読み込んでいます...")
def get_preview(path: str, mtime: float, roi: Roi) -> np.ndarray:
    return best_preview_frame(path, roi)


@st.cache_data(show_spinner="選んだ光源の明るさを取り出しています(長い動画では数分かかります)...")
def get_brightness(path: str, mtime: float, roi: Roi) -> np.ndarray:
    _, values = extract_brightness(path, roi)
    return values


# --- 画面 -------------------------------------------------------------------


def _browse() -> None:
    """「参照…」を押したとき。入力欄が作られる前(再実行の前)に呼ばれるので、ここで欄の値を差し替える。"""
    chosen = pick_video_file(initial_dir_for(st.session_state.get("video_path", "")))
    if chosen:
        st.session_state["video_path"] = chosen


def main() -> None:
    st.set_page_config(page_title="光モールス信号AI翻訳", page_icon="🔦", layout="wide")
    st.title("🔦 光モールス信号AI翻訳")
    st.caption("スマホ(R10)で撮った光の点滅の動画から、点滅している光源を自動で見つけ、和文モールス信号を読み取ります。")

    model_path = MODELS_DIR / "best_model.pt"
    if not model_path.exists():
        st.error(f"学習済みモデルが見つかりません: {model_path}")
        return
    model, labels = get_model(model_path.stat().st_mtime)

    with st.sidebar:
        st.header("設定")
        low_conf = st.slider("低確信度しきい値", 0.0, 1.0, DEFAULT_LOW_CONFIDENCE, 0.05,
                             help="確信度がこの値未満の文字を「要確認」として色を付けます。")

    st.subheader("1. 動画を選ぶ")
    col_path, col_browse = st.columns([6, 1], vertical_alignment="bottom")
    path_input = col_path.text_input(
        "動画ファイルの場所(例: C:\\Users\\daizo\\OneDrive\\光点滅テスト\\...\\xxx.mp4)", key="video_path",
        help="文章の動画は数GBになるため、「参照…」で選ぶか、ファイルの場所を入力してください。大きさの上限はありません。"
        "OneDrive のファイルも使えます(読み込み時にダウンロードされます)。")
    col_browse.button("参照…", on_click=_browse, use_container_width=True,
                      help="ファイルを選ぶ画面を開きます(ブラウザの手前に出ます)。")
    uploaded = st.file_uploader("または動画をアップロード(1GBまで)", type=["mp4", "mov"])
    path = ""
    if uploaded is not None:
        tmp = Path(tempfile.gettempdir()) / f"light_upload_{uploaded.name}"
        if not tmp.exists() or tmp.stat().st_size != uploaded.size:
            tmp.write_bytes(uploaded.getvalue())
        path = str(tmp)
    elif path_input.strip():
        path = path_input.strip().strip('"')
    if not path:
        st.info("動画の場所を入力するか、動画をアップロードしてください。")
        return
    if not Path(path).exists():
        st.error(f"ファイルが見つかりません: {path}")
        return
    mtime = Path(path).stat().st_mtime

    try:
        rois, fps, n_frames = get_candidates(path, mtime)
    except Exception as exc:  # noqa: BLE001
        st.error(f"光源を探せませんでした: {exc}")
        return
    if not rois:
        st.warning("点滅している光源が見つかりませんでした。")
        return

    st.subheader("2. 光源を確かめる(フォーカス)")
    options = [f"{k + 1}番(点数 {r.score:.0f})" for k, r in enumerate(rois)]
    choice = st.radio("読み取る光源", options, index=0, horizontal=True,
                      help="1番が自動で選んだ光源です。違う光を選んでいる場合は、ほかの番号を選んでください。")
    selected = options.index(choice)
    roi = rois[selected]
    preview = get_preview(path, mtime, roi)
    col1, col2 = st.columns([3, 1])
    col1.image(draw_candidates(preview, rois, selected), caption="緑の枠 = 読み取る光源、黄色の枠 = ほかの候補", use_container_width=True)
    col2.image(zoom_crop(preview, roi), caption=f"選んだ光源の拡大({roi.x}, {roi.y})", use_container_width=True)
    col2.caption(f"動画: {n_frames}コマ、{fps:.1f}コマ/秒、約{n_frames / fps:.0f}秒")

    values = get_brightness(path, mtime, roi)
    result = recognize(values, model, labels)
    if not result.segments:
        st.warning("点滅を区切れませんでした。別の光源を選んでみてください。")
        return

    st.subheader("3. 読み取り結果")
    st.markdown(render_text_html(result, low_conf), unsafe_allow_html=True)
    review = sum(1 for s in result.segments if s.confidence < low_conf)
    if review:
        st.caption("背景が橙色の文字は確信度が低く、要確認です。")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("文字数(濁音は1文字)", len(result.text))
    m2.metric("短点の長さ", f"{result.unit_frames / fps * 1000:.0f} ms")
    m3.metric("平均確信度", f"{np.mean([s.confidence for s in result.segments]):.2f}")
    m4.metric("要確認の区間", review)
    st.download_button("読み取った文章を保存(.txt)", result.text, file_name=f"{Path(path).stem}_読み取り.txt")

    st.subheader("明るさの波形")
    total = len(values) / fps
    if total > 30:
        start = st.slider("表示する範囲の始まり[秒]", 0.0, max(0.0, total - 20), 0.0, 1.0)
        window = (start, start + 20)
    else:
        window = None
    st.pyplot(plot_trace(values, fps, result, window))
    plt.close("all")

    st.subheader("文字ごとの詳細")
    table = build_char_table(result, fps, low_conf)
    only_review = st.checkbox("要確認の文字のみ表示", value=False)
    st.dataframe(table[table["要確認"] != ""] if only_review else table, hide_index=True, use_container_width=True,
                 column_config={"確信度": st.column_config.ProgressColumn("確信度", min_value=0.0, max_value=1.0, format="%.2f")})


if __name__ == "__main__":
    main()
