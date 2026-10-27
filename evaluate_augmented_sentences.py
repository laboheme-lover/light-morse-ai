"""文章の動画(evaluate_sentences で効果なしで測った20本)に映像の段階の悪条件(light_augment)をかけ、
光源の自動検知から文章の読み取りまでを通して測る。

1本 × 1条件ごとに
  (1) 元の動画の各コマに効果をかけた動画を一時ファイルに作る(1文字の評価 evaluate_augmented_singles と同じ方法)
  (2) 光源を自動で見つけ(最初の60秒)、動画全体の明るさの列を取り出す
  (3) 短点の長さも効果をかけた列から測り直し(1文字の評価では固定の 9.00コマを使った)、文章として読み取る
  (4) 正解と比べる(evaluate_sentences と同じ数え方)
条件は1文字の評価と同じ6通り。効果の強さは seed で1本ごとに抽選し、その動画の中では一定。
本物のランプの位置は、効果なしの評価で見つけた ROI を使う(効果の光源をそこに重ねない・検知の当たり判定に使う)。
明るさの列は data/aug_sentences に保存する(モデルを変えたときに動画を作り直さずに読み直せる)。

使い方: python evaluate_augmented_sentences.py <動画のフォルダ> ... [--workers 12]
結果: evaluation/augmented_sentences_report.csv(1本×1条件ごと)
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from evaluate_augmented_singles import CONDITION_SETS

FIELDS = ["file", "conditions", "clean_roi", "roi", "roi_hit", "frames", "unit_frames", "ground_truth_len",
          "predicted_len", "similarity", "char_match_rate", "char_substitution_rate", "char_insertion_rate",
          "char_deletion_rate", "seconds", "error", "predicted_text", "ground_truth"]

_MODEL = None


def _init(models_dir: str) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    import cv2
    import torch

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    from evaluate_real_singles import load_model

    global _MODEL
    _MODEL = load_model(Path(models_dir))


def make_augmented_video(src: str, dst: str, conditions: tuple[str, ...], seed: int,
                         avoid: tuple[int, int, int, int]) -> int:
    """src の各コマに効果をかけて dst に書く。書いたコマ数を返す。"""
    import cv2

    from light_augment import SceneAugmenter

    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aug = SceneAugmenter(effects=conditions, shape=(h, w), fps=fps, seed=seed, avoid=avoid)
    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(aug.apply(frame, i))
        i += 1
    cap.release()
    writer.release()
    return i


def _job(job: dict) -> dict:
    from evaluate_sentences import compare_texts, normalize_ground_truth
    from light_roi import detect_signal_roi, extract_brightness
    from light_sequence import recognize

    started = time.time()
    video = Path(job["video"])
    cond = "+".join(job["conditions"])
    row = {k: "" for k in FIELDS}
    row.update(file=video.name, conditions=cond, clean_roi=job["clean_roi"])
    cache = Path(job["cache_dir"]) / f"{video.stem}_{cond}.npz"
    tmp = Path(tempfile.gettempdir()) / f"augsent_{video.stem}_{cond}_{os.getpid()}.mp4"
    try:
        avoid = tuple(int(v) for v in job["clean_roi"].split(","))
        if cache.exists():
            d = np.load(cache)
            values, roi_text = d["v"], str(d["roi"])
        else:
            make_augmented_video(str(video), str(tmp), job["conditions"], job["seed"], avoid)
            roi = detect_signal_roi(str(tmp))
            _, values = extract_brightness(str(tmp), roi)
            roi_text = f"{roi.x},{roi.y},{roi.w},{roi.h}"
            np.savez_compressed(cache, v=values, roi=roi_text)
        x, y, w, h = (int(v) for v in roi_text.split(","))
        cx, cy = avoid[0] + avoid[2] / 2, avoid[1] + avoid[3] / 2
        model, labels = _MODEL
        result = recognize(values, model, labels)
        gt = normalize_ground_truth(video.with_suffix(".txt").read_text(encoding="utf-8-sig"))
        cmp = compare_texts(result.text, gt)
        row.update({"roi": roi_text, "roi_hit": x <= cx <= x + w and y <= cy <= y + h, "frames": len(values),
                    "unit_frames": round(result.unit_frames, 2), "ground_truth_len": len(gt),
                    "predicted_len": len(result.text), "predicted_text": result.text, "ground_truth": gt,
                    **{k: round(v, 4) for k, v in cmp.items()}})
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        tmp.unlink(missing_ok=True)
    row["seconds"] = round(time.time() - started, 1)
    return row


def main() -> None:
    from multiprocessing import Pool

    parser = argparse.ArgumentParser(description="効果をかけた文章の動画で、検知から読み取りまでを通して測る")
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--clean-report", default="evaluation/sentences_report.csv",
                        help="効果なしの評価の結果(対象の動画と本物のランプの ROI)")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--cache-dir", default="data/aug_sentences")
    parser.add_argument("--output", default="evaluation/augmented_sentences_report.csv")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    with open(args.clean_report, encoding="utf-8") as f:
        clean = {r["file"]: r for r in csv.DictReader(f) if not r["error"]}
    videos = {}
    for folder in args.folders:
        for v in sorted(Path(folder).glob("*_sentence.mp4")):
            if v.name in clean and v.with_suffix(".txt").exists():
                videos.setdefault(v.name, v)
    missing = sorted(set(clean) - set(videos))
    if missing:
        print(f"動画が見つからないもの: {missing}", flush=True)
    jobs = []
    for k, name in enumerate(sorted(videos)):
        for c, conditions in enumerate(CONDITION_SETS):
            jobs.append({"video": str(videos[name]), "conditions": conditions, "seed": 1000 * k + c,
                         "clean_roi": clean[name]["roi"], "frames": int(clean[name]["frames"]),
                         "cache_dir": args.cache_dir})
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            done = {(r["file"], r["conditions"]) for r in csv.DictReader(f) if not r["error"]}
    todo = [j for j in jobs if (Path(j["video"]).name, "+".join(j["conditions"])) not in done]
    todo.sort(key=lambda j: -j["frames"])  # 長い動画から始めて、最後に1本だけ残って待つ時間を減らす
    print(f"対象 {len(jobs)}件 / 今回 {len(todo)}件", flush=True)
    new = not out.exists()
    with open(out, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        with Pool(args.workers, initializer=_init, initargs=(args.models_dir,)) as pool:
            for n, row in enumerate(pool.imap_unordered(_job, todo), start=1):
                writer.writerow(row)
                f.flush()
                print(f"[{n}/{len(todo)}] {row['file']} {row['conditions']} 検知 {row['roi_hit']} "
                      f"一致率 {row['char_match_rate']} 短点 {row['unit_frames']}コマ {row['seconds']}秒 {row['error']}",
                      flush=True)


if __name__ == "__main__":
    main()
