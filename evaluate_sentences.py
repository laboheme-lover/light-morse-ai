"""文章を送った実際の動画で、光源の自動検知から文章の読み取りまでを通して測る。

対象: <フォルダ>/*_sentence.mp4 と同じ名前の .txt(正解)。光点滅テストの original44(実録音の文章44件と同じ文)、
original20(電気通信術の電文20件と同じ文)など。
1本ごとに
  (1) 光源を自動で見つけ(最初の60秒)、動画全体の明るさの列を取り出す。列は保存し、2回目からは動画を読まない
  (2) light_sequence.recognize で読み取る(短点の長さは動画全体から測る)
  (3) 正解と比べる。正解は空白を除き、「」」は「。」(同じ符号)とし、濁点は1文字に結合する。
      括弧「（」「）」は送信されているので正解に残す(モデルは括弧を知らないので、誤りとして数えられる)
結果: evaluation/sentences_report.csv
"""

from __future__ import annotations

import argparse
import csv
import difflib
import os
import time
from pathlib import Path

import numpy as np

FIELDS = ["file", "roi", "frames", "unit_frames", "ground_truth_len", "predicted_len", "segments", "similarity",
          "char_match_rate", "char_substitution_rate", "char_insertion_rate", "char_deletion_rate", "seconds",
          "error", "predicted_text", "ground_truth"]


def normalize_ground_truth(text: str) -> str:
    from morse_table import compose_dakuten

    text = "".join(text.split()).replace("」", "。")
    return compose_dakuten(text)


def compare_texts(predicted: str, ground_truth: str) -> dict:
    """音の evaluate_real_samples.compare_texts と同じ数え方(autojunk=False、分母は正解の文字数)。"""
    matcher = difflib.SequenceMatcher(None, ground_truth, predicted, autojunk=False)
    matched = substituted = inserted = deleted = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        g, p = i2 - i1, j2 - j1
        if tag == "equal":
            matched += g
        elif tag == "replace":
            k = min(g, p)
            substituted += k
            deleted += g - k
            inserted += p - k
        elif tag == "delete":
            deleted += g
        elif tag == "insert":
            inserted += p
    n = max(len(ground_truth), 1)
    return {"similarity": matcher.ratio(), "char_match_rate": matched / n, "char_substitution_rate": substituted / n,
            "char_insertion_rate": inserted / n, "char_deletion_rate": deleted / n}


ONLINE_ONLY = 0x400000 | 0x1000  # Windows のファイル属性: 開くとダウンロードされる(OneDrive の「オンラインのみ」)


def ensure_local(video: Path) -> None:
    """OneDrive の「オンラインのみ」のファイルを、先頭から最後まで読み通してダウンロードを済ませる。

    OpenCV でいきなり開くと、数GBのダウンロードが終わる前に読み込みが時間切れになる(約17分で失敗した)。
    普通のファイル読み込みはダウンロードが終わるまで待つので、先に読み通しておく。
    """
    if not getattr(os.stat(video), "st_file_attributes", 0) & ONLINE_ONLY:
        return
    with open(video, "rb") as f:
        while f.read(64 * 1024 * 1024):
            pass


def trace_for(video: Path, cache_dir: Path) -> tuple[np.ndarray, str]:
    from light_roi import detect_signal_roi, extract_brightness

    cache = cache_dir / f"{video.stem}_trace.npz"
    if cache.exists():
        d = np.load(cache)
        return d["v"], str(d["roi"])
    ensure_local(video)
    roi = detect_signal_roi(str(video))
    t, v = extract_brightness(str(video), roi)
    roi_text = f"{roi.x},{roi.y},{roi.w},{roi.h}"
    np.savez_compressed(cache, t=t, v=v, roi=roi_text)
    return v, roi_text


def _job(job: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    from evaluate_real_singles import load_model
    from light_sequence import recognize

    started = time.time()
    video = Path(job["video"])
    row = {k: "" for k in FIELDS}
    row["file"] = video.name
    try:
        values, roi = trace_for(video, Path(job["cache_dir"]))
        model, labels = load_model(Path(job["models_dir"]))
        result = recognize(values, model, labels)
        gt = normalize_ground_truth(video.with_suffix(".txt").read_text(encoding="utf-8-sig"))
        cmp = compare_texts(result.text, gt)
        row.update({"roi": roi, "frames": len(values), "unit_frames": round(result.unit_frames, 2),
                    "ground_truth_len": len(gt), "predicted_len": len(result.text), "segments": len(result.segments),
                    "predicted_text": result.text, "ground_truth": gt, **{k: round(v, 4) for k, v in cmp.items()}})
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = round(time.time() - started, 1)
    return row


def main() -> None:
    from multiprocessing import Pool

    parser = argparse.ArgumentParser(description="文章の動画で、検知から読み取りまでを通して測る")
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--cache-dir", default="data/real_sentences")
    parser.add_argument("--output", default="evaluation/sentences_report.csv")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    videos = {}
    for folder in args.folders:  # 同じ名前の動画が複数のフォルダにある場合は、最初のものを使う
        for v in sorted(Path(folder).glob("*_sentence.mp4")):
            if v.with_suffix(".txt").exists():
                videos.setdefault(v.name, v)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            done = {r["file"] for r in csv.DictReader(f) if not r["error"]}
    jobs = [{"video": str(v), "cache_dir": str(cache), "models_dir": args.models_dir}
            for name, v in sorted(videos.items()) if name not in done]
    print(f"対象 {len(videos)}本 / 今回 {len(jobs)}本", flush=True)
    new = not out.exists()
    with open(out, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        with Pool(args.workers) as pool:
            for n, row in enumerate(pool.imap_unordered(_job, jobs), start=1):
                writer.writerow(row)
                f.flush()
                print(f"[{n}/{len(jobs)}] {row['file']} 一致率 {row['char_match_rate']} 短点 {row['unit_frames']}コマ "
                      f"{row['seconds']}秒 {row['error']}", flush=True)


if __name__ == "__main__":
    main()
