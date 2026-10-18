"""実際の動画(1文字ずつ65本)で、学習したモデルの正解率を測る。

measure_clips.py が保存した明るさの列(data/real_single65/*_trace.npz)を使う(動画は読み直さない)。
短点の長さは、文章の動画と同じく「録画全体」から測る: 65本すべての点灯の長さを集め、短いほうから15%の
位置の値を短点の長さとする(音の audio_preprocessing.estimate_dot_seconds と同じ考え方)。
1本だけでは「ヨ」(--)と「ヌ」(....)のように短点の長さが分からない文字があるため。

「」」は「。」として採点する(light_table.class_for_label)。括弧「（」「）」も学習した文字なので採点に含める。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from light_features import normalize, prepare_input
from light_model import MorseLightNet
from light_table import class_for_label

DOT_PERCENTILE = 15


def on_run_lengths(values: np.ndarray) -> list[int]:
    """明るさの列の、点灯が続いたコマ数の一覧。"""
    on = normalize(values) > 0.5
    lengths, count = [], 0
    for b in on:
        if b:
            count += 1
        elif count:
            lengths.append(count)
            count = 0
    if count:
        lengths.append(count)
    return lengths


def estimate_unit_frames(traces: list[np.ndarray]) -> float:
    lengths = [n for v in traces for n in on_run_lengths(v)]
    return float(np.percentile(lengths, DOT_PERCENTILE))


def load_model(models_dir: Path) -> tuple[MorseLightNet, list[str]]:
    mapping = json.loads((models_dir / "label_mapping.json").read_text(encoding="utf-8"))
    state = torch.load(models_dir / "best_model.pt", weights_only=False)
    model = MorseLightNet(num_classes=len(mapping["labels"]))
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, mapping["labels"]


def predict(model, labels, values: np.ndarray, unit_frames: float, top_k: int = 3) -> list[tuple[str, float]]:
    x = torch.from_numpy(prepare_input(values, unit_frames))[None, None, :]
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0].numpy()
    order = np.argsort(prob)[::-1][:top_k]
    return [(labels[i], float(prob[i])) for i in order]


def main() -> None:
    parser = argparse.ArgumentParser(description="実際の動画(1文字65本)で正解率を測る")
    parser.add_argument("--traces", default="data/real_single65")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--output", default="evaluation/real_single65_report.csv")
    args = parser.parse_args()

    model, labels = load_model(Path(args.models_dir))
    files = sorted(Path(args.traces).glob("single_*_trace.npz"))
    data = [(f.stem.replace("_trace", ""), np.load(f)) for f in files]
    unit = estimate_unit_frames([d["v"] for _, d in data])
    print(f"録画全体から測った短点の長さ: {unit:.2f}コマ")
    rows = []
    for clip_id, d in data:
        label = str(d["label"])
        target = class_for_label(label)
        top = predict(model, labels, d["v"], unit)
        rows.append({"id": clip_id, "label": label, "target": target or "", "pred": top[0][0],
                     "confidence": round(top[0][1], 4), "top3": " / ".join(f"{c}({p:.2f})" for c, p in top),
                     "correct": (top[0][0] == target) if target else ""})
    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")
    scored = df[df["target"] != ""]
    print(f"正解率: {(scored['correct'] == True).sum()} / {len(scored)}(「」」は「。」として採点)")
    wrong = scored[scored["correct"] != True]
    if len(wrong):
        print(wrong[["id", "label", "pred", "top3"]].to_string(index=False))


if __name__ == "__main__":
    main()
