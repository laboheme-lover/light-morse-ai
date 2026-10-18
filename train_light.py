"""光のモールス信号のモデルを学習する。音の train.py にあたる。

学習データ: generate_light_dataset.py が作った data/dataset(65文字 × 300件を 学習70% / 検証15% / テスト15% に分けたもの)。
入力は light_features.prepare_input(推論と共通)。短点の長さは本来、動画全体から測った値を使うので、
学習では「本当の短点の長さ」に ±15% の測り違いを加えて渡す(測り違いがあっても読めるようにするため)。
検証・テストでは、件ごとに決まった測り違い(index から作る)を使い、毎回同じ条件で比べる。

保存するもの(models/)
  best_model.pt       検証の正解率が最も高かったエポックのモデル
  label_mapping.json  文字 ⇔ 番号の対応と入力の長さ
  train_history.json  エポックごとの損失と正解率
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from generate_light_dataset import load_traces
from light_features import INPUT_LENGTH, prepare_input
from light_model import MorseLightNet, count_parameters
from light_table import CLASS_CODES

UNIT_ESTIMATE_ERROR = 0.15  # 短点の長さの測り違いの幅(±15%)


class LightTraceDataset(Dataset):
    def __init__(self, traces: list[np.ndarray], meta: pd.DataFrame, label_to_index: dict[str, int], train: bool):
        self.traces = traces
        self.meta = meta.reset_index(drop=True)
        self.labels = [label_to_index[c] for c in self.meta["label"]]
        self.train = train
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, i: int):
        row = self.meta.iloc[i]
        unit_frames = float(row["unit_sec"]) * float(row["fps"])
        if self.train:
            error = self.rng.uniform(-UNIT_ESTIMATE_ERROR, UNIT_ESTIMATE_ERROR)
        else:  # 検証・テストは件ごとに決まった測り違い
            error = np.random.default_rng(int(row["index"])).uniform(-UNIT_ESTIMATE_ERROR, UNIT_ESTIMATE_ERROR)
        x = prepare_input(self.traces[int(row["index"])], unit_frames * (1 + error))
        return torch.from_numpy(x)[None, :], self.labels[i]


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            count += len(y)
    return total_loss / count, correct / count


def evaluate(model, loader) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in loader:
            preds.append(model(x).argmax(1).numpy())
    return np.concatenate(preds)


def main() -> None:
    parser = argparse.ArgumentParser(description="光のモールス信号のモデルを学習する")
    parser.add_argument("--data", default="data/dataset")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    data = Path(args.data)
    traces = load_traces(data)
    meta = pd.read_csv(data / "metadata.csv", encoding="utf-8", keep_default_na=False)
    labels = list(CLASS_CODES)
    label_to_index = {c: i for i, c in enumerate(labels)}
    sets = {s: LightTraceDataset(traces, meta[meta["split"] == s], label_to_index, train=(s == "train"))
            for s in ("train", "val", "test")}
    loaders = {s: DataLoader(d, batch_size=args.batch_size, shuffle=(s == "train")) for s, d in sets.items()}

    model = MorseLightNet(num_classes=len(labels))
    print(f"パラメータ数: {count_parameters(model):,}  学習 {len(sets['train'])} / 検証 {len(sets['val'])} / テスト {len(sets['test'])}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "label_mapping.json").write_text(
        json.dumps({"labels": labels, "input_length": INPUT_LENGTH}, ensure_ascii=False, indent=1), encoding="utf-8")

    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, optimizer)
        va_loss, va_acc = run_epoch(model, loaders["val"], criterion)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
        if va_acc > best:
            best = va_acc
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_acc": va_acc, "labels": labels},
                       models_dir / "best_model.pt")
        (models_dir / "train_history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
        print(f"エポック {epoch:2d}: 学習 {tr_acc:.4f}(損失 {tr_loss:.3f}) 検証 {va_acc:.4f}(損失 {va_loss:.3f}) "
              f"{time.time() - t0:.0f}秒", flush=True)

    # テストデータでの最終評価(最良のモデル)
    state = torch.load(models_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    test_meta = sets["test"].meta
    pred = evaluate(model, loaders["test"])
    test_meta = test_meta.assign(pred=[labels[i] for i in pred])
    test_meta["correct"] = test_meta["pred"] == test_meta["label"]
    print(f"最良エポック {state['epoch']}(検証 {state['val_acc']:.4f})  テスト正解率 {test_meta['correct'].mean():.4f}")
    by_cond = test_meta.groupby("conditions")["correct"].agg(["mean", "size"]).sort_values("mean")
    print("条件ごとのテスト正解率(低い順、上位10):")
    print(by_cond.head(10).to_string())
    test_meta.to_csv(models_dir / "test_predictions.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
