"""実際の動画(1文字65本)に映像の段階の悪条件(light_augment)をかけ、光源の自動検知から文字の判定までを通して測る。

1本 × 1条件ごとに
  (1) 元の動画の各コマに効果をかけた動画を一時ファイルに作る
  (2) 光源を自動で見つけ(light_roi)、明るさの列を取り出す
  (3) 短点の長さを決めてモデルで判定する
短点の長さは、文章の動画と同じく録画全体から測る想定なので、効果をかけていない65本から測った値
(evaluate_real_singles と同じ 9.00コマ)を使う。悪条件の下でそれを測り直す影響は、文章の評価で別に見る。

使い方: python evaluate_augmented_singles.py <動画のフォルダ> [--workers 8]
結果: evaluation/augmented_singles_report.csv(1本×1条件ごと)
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
import time
from pathlib import Path

CONDITION_SETS = [("dusk",), ("fog",), ("rain",), ("lights",), ("noise",), ("dusk", "rain", "lights", "noise")]
TRUE_ROI = (930, 550, 40, 40)  # 記録にあるランプの位置(効果の光源をここに重ねないため)
FIELDS = ["id", "label", "target", "conditions", "roi", "roi_hit", "pred", "confidence", "correct", "seconds", "error"]

_MODEL = None


def _init(models_dir: str) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    from evaluate_real_singles import load_model

    global _MODEL
    _MODEL = load_model(Path(models_dir))


def _job(job: dict) -> dict:
    import cv2

    from evaluate_real_singles import predict
    from light_augment import SceneAugmenter
    from light_roi import detect_signal_roi, extract_brightness
    from light_table import class_for_label

    started = time.time()
    row = {k: "" for k in FIELDS}
    row.update(id=Path(job["video"]).stem, label=job["label"], conditions="+".join(job["conditions"]))
    target = class_for_label(job["label"])
    row["target"] = target or ""
    tmp = Path(tempfile.gettempdir()) / f"aug_{row['id']}_{row['conditions']}_{os.getpid()}.mp4"
    try:
        cap = cv2.VideoCapture(job["video"])
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        aug = SceneAugmenter(effects=job["conditions"], shape=(h, w), fps=fps, seed=job["seed"], avoid=TRUE_ROI)
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(aug.apply(frame, i))
            i += 1
        cap.release()
        writer.release()
        roi = detect_signal_roi(str(tmp))
        _, values = extract_brightness(str(tmp), roi)
        cx, cy = TRUE_ROI[0] + TRUE_ROI[2] / 2, TRUE_ROI[1] + TRUE_ROI[3] / 2
        row["roi"] = f"{roi.x},{roi.y},{roi.w},{roi.h}"
        row["roi_hit"] = roi.x <= cx <= roi.x + roi.w and roi.y <= cy <= roi.y + roi.h
        model, labels = _MODEL
        top = predict(model, labels, values, job["unit_frames"])
        row["pred"], row["confidence"] = top[0][0], round(top[0][1], 4)
        row["correct"] = (top[0][0] == target) if target else ""
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        tmp.unlink(missing_ok=True)
    row["seconds"] = round(time.time() - started, 1)
    return row


def main() -> None:
    from multiprocessing import Pool

    parser = argparse.ArgumentParser(description="効果をかけた実際の動画で、検知から判定までを通して測る")
    parser.add_argument("video_dir")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--output", default="evaluation/augmented_singles_report.csv")
    parser.add_argument("--unit-frames", type=float, default=9.0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    jobs = []
    for k, video in enumerate(sorted(Path(args.video_dir).glob("single_*.mp4"))):
        label = video.with_suffix(".txt").read_text(encoding="utf-8-sig").strip()
        for c, conditions in enumerate(CONDITION_SETS):
            jobs.append({"video": str(video), "label": label, "conditions": conditions, "seed": 1000 * k + c,
                         "unit_frames": args.unit_frames})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            done = {(r["id"], r["conditions"]) for r in csv.DictReader(f) if not r["error"]}
    todo = [j for j in jobs if (Path(j["video"]).stem, "+".join(j["conditions"])) not in done]
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
                print(f"[{n}/{len(todo)}] {row['id']} {row['conditions']} 検知 {row['roi_hit']} 判定 {row['pred']} "
                      f"{row['correct']} {row['error']}", flush=True)


if __name__ == "__main__":
    main()
