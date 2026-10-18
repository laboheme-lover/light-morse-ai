"""実際の動画(1文字ずつの65本)から明るさの波形を取り出し、合成の元になる値を測る。

各動画について
  (1) 点滅する光源を自動で見つけ(light_roi.detect_signal_roi)、明るさの波形を取り出す
  (2) 点灯・消灯を2値にし、点灯/消灯それぞれの長さ(コマ数)の並びを求める
  (3) 正解の文字の符号(短点・長点の並び)と比べ、短点・長点・間の長さを測る
結果は 1本ずつの表(CSV)と、波形そのもの(npz)に保存する。波形を保存しておけば、
あとで測り方を変えるときに動画(1本約80MB)を読み直さずに済む。

使い方: python measure_clips.py <動画のフォルダ> <出力フォルダ>
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from light_roi import detect_signal_roi, extract_brightness
from light_table import code_for_label

FIELDS = ["id", "label", "code", "roi", "roi_score", "frames", "fps", "off_level", "on_level", "contrast",
          "runs_on", "runs_off", "code_from_light", "code_match", "dot_frames", "dash_frames", "gap_frames",
          "dot_ms", "dash_ms", "gap_ms", "edge_frames"]


def runs(binary: np.ndarray) -> list[tuple[bool, int]]:
    """0/1 の列を (点灯か, 続いたコマ数) の並びにする。"""
    out = []
    start = 0
    for i in range(1, len(binary) + 1):
        if i == len(binary) or binary[i] != binary[start]:
            out.append((bool(binary[start]), i - start))
            start = i
    return out


def measure(video: Path, label: str, fps_hint: float = 30.3) -> tuple[dict, np.ndarray, np.ndarray]:
    roi = detect_signal_roi(str(video))
    t, v = extract_brightness(str(video), roi)
    fps = (len(t) - 1) / t[-1] if len(t) > 1 and t[-1] > 0 else fps_hint
    off, on = np.percentile(v, 5), np.percentile(v, 95)
    binary = (v > (off + on) / 2).astype(int)
    seq = runs(binary)
    # 前後の消灯(信号の外)を除いた、信号の部分だけの並び
    while seq and not seq[0][0]:
        seq.pop(0)
    while seq and not seq[-1][0]:
        seq.pop()
    on_runs = [n for is_on, n in seq if is_on]
    off_runs = [n for is_on, n in seq if not is_on]
    code = code_for_label(label)
    row = {
        "id": video.stem, "label": label, "code": code, "roi": json.dumps([roi.x, roi.y, roi.w, roi.h]),
        "roi_score": round(roi.score, 1), "frames": len(v), "fps": round(fps, 3),
        "off_level": round(float(off), 1), "on_level": round(float(on), 1), "contrast": round(float(on - off), 1),
        "runs_on": " ".join(map(str, on_runs)), "runs_off": " ".join(map(str, off_runs)),
    }
    # 点灯の長さを、短い群(短点)と長い群(長点)に分ける。境目は「最短の2倍」(短点1単位・長点3単位の中間)
    if on_runs:
        shortest = min(on_runs)
        light_code = "".join("-" if n >= 2 * shortest else "." for n in on_runs)
        # 短点しか無い文字(ヌ・ヘ・ラ 等)は、最短の2倍の基準では長点を区別できないので、正解の符号の数で確かめる
        row["code_from_light"] = light_code
        row["code_match"] = (light_code == code) if code else ""
        if code and len(on_runs) == len(code):
            dots = [n for n, c in zip(on_runs, code) if c == "."]
            dashes = [n for n, c in zip(on_runs, code) if c == "-"]
            row["dot_frames"] = round(float(np.mean(dots)), 2) if dots else ""
            row["dash_frames"] = round(float(np.mean(dashes)), 2) if dashes else ""
            row["gap_frames"] = round(float(np.mean(off_runs)), 2) if off_runs else ""
            for key in ("dot", "dash", "gap"):
                if row[f"{key}_frames"] != "":
                    row[f"{key}_ms"] = round(row[f"{key}_frames"] / fps * 1000, 1)
    # 立ち上がり・立ち下がりで、点灯と消灯の中間の明るさになっているコマの数(にじみ・露出の影響)
    mid = (v > off + 0.2 * (on - off)) & (v < on - 0.2 * (on - off))
    row["edge_frames"] = int(mid.sum())
    return row, t, v


def main() -> None:
    video_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "clip_measurements.csv"
    done = set()
    if report.exists():
        with open(report, encoding="utf-8") as f:
            done = {r["id"] for r in csv.DictReader(f)}
    new_file = not report.exists()
    with open(report, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for video in sorted(video_dir.glob("single_*.mp4")):
            if video.stem in done:
                continue
            label = video.with_suffix(".txt").read_text(encoding="utf-8-sig").strip()
            row, t, v = measure(video, label)
            np.savez_compressed(out_dir / f"{video.stem}_trace.npz", t=t, v=v, label=label)
            writer.writerow({k: row.get(k, "") for k in FIELDS})
            f.flush()
            print(video.stem, label, row["code"], row.get("code_from_light"), row.get("code_match"),
                  "dot", row.get("dot_ms"), "dash", row.get("dash_ms"), "gap", row.get("gap_ms"), flush=True)


if __name__ == "__main__":
    main()
