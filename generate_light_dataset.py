"""光のモールス信号の学習データ(ランプの明るさの波形)を作る。音の generate_dataset.py にあたる。

65文字(音の63文字 + 括弧「（」「）」)× 各300件 = 19,500件。1件 = 1文字分の明るさの列(1コマに1つの値、30.3コマ/秒前後)。
学習 70% / 検証 15% / テスト 15% に、文字ごとに分ける(音と同じ割合)。

1件ごとの条件の抽選(SynthParams と悪条件)
  - 速さ: 短点 0.15〜0.6秒(実際のテスト動画は 0.3秒)。露光・コマの間隔・ランプの点き方は実測に合わせた範囲
  - 悪条件: 20% は雑音なし。残りは dusk / fog / rain / lights / camera から1〜3個を選んで重ねる
    (音で「雑音なし20%・白色雑音40%・環境音40%」とした考え方に合わせ、楽なものから厳しいものまで混ぜる)

出力(<output_root>)
  traces.npz    すべての波形を1本につないだ values と、各件の始まりの位置 offsets(件数+1個)
  metadata.csv  1件ごとの index・split・文字・条件(速さ・悪条件・シードなど)

実際の動画(1文字65本)は合成の調整に使っただけで、このデータには入れない(確認用に取っておく)。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from light_synth import CONDITIONS, add_conditions, sample_params, synthesize_char
from light_table import CLASS_CODES

CLEAN_RATIO = 0.2
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
METADATA_FIELDS = ["index", "split", "label", "unit_sec", "fps", "exposure", "rise_ms", "fall_ms", "conditions",
                   "frames", "seed"]


def assign_splits(n: int, rng: np.random.Generator) -> list[str]:
    """n件を 学習/検証/テスト に割り当てる(並びはランダム)。"""
    n_train = int(round(n * SPLIT_RATIOS["train"]))
    n_val = int(round(n * SPLIT_RATIOS["val"]))
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
    rng.shuffle(splits)
    return splits


def sample_conditions(rng: np.random.Generator) -> tuple[str, ...]:
    if rng.random() < CLEAN_RATIO:
        return ()
    k = int(rng.integers(1, 4))
    return tuple(sorted(rng.choice(CONDITIONS, size=k, replace=False)))


def generate(output_root: Path, per_class: int, seed: int) -> Path:
    master = np.random.default_rng(seed)
    output_root.mkdir(parents=True, exist_ok=True)
    values: list[np.ndarray] = []
    rows: list[dict] = []
    for label in CLASS_CODES:
        splits = assign_splits(per_class, master)
        for split in splits:
            s = int(master.integers(0, 2**31 - 1))
            rng = np.random.default_rng(s)
            trace, p = synthesize_char(label, rng, sample_params(rng))
            conditions = sample_conditions(rng)
            if conditions:
                trace = add_conditions(trace, conditions, rng, fps=p.fps)
            values.append(trace.astype(np.float32))
            rows.append({
                "index": len(rows), "split": split, "label": label, "unit_sec": round(p.unit, 4),
                "fps": round(p.fps, 3), "exposure": round(p.exposure, 3), "rise_ms": round(p.rise * 1000, 1),
                "fall_ms": round(p.fall * 1000, 2), "conditions": "+".join(conditions) or "clean",
                "frames": len(trace), "seed": s,
            })
    offsets = np.r_[0, np.cumsum([len(v) for v in values])].astype(np.int64)
    np.savez_compressed(output_root / "traces.npz", values=np.concatenate(values), offsets=offsets)
    metadata = output_root / "metadata.csv"
    with open(metadata, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return metadata


def load_traces(output_root: Path) -> list[np.ndarray]:
    """traces.npz を1件ずつの波形のリストに戻す。"""
    data = np.load(output_root / "traces.npz")
    values, offsets = data["values"], data["offsets"]
    return [values[a:b] for a, b in zip(offsets[:-1], offsets[1:])]


def main() -> None:
    parser = argparse.ArgumentParser(description="光のモールス信号の学習データを作る")
    parser.add_argument("--output-root", default="data/dataset")
    parser.add_argument("--per-class", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = generate(Path(args.output_root), args.per_class, args.seed)
    print(f"作成しました: {path}")


if __name__ == "__main__":
    main()
