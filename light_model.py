"""光のモールス信号(1文字分の明るさの列)を読み取るモデル。

音のモデル(morse-ai の MorseCRNN: 畳み込み4段+双方向GRU)と同じ考え方の作りを、1次元の列に合わせて小さくした。
  入力: (件数, 1, 144) … light_features.prepare_input の出力(1単位 = 6点、24単位)
  畳み込み3段: 点灯・消灯の短い形(短点・長点・間)をとらえる
  双方向GRU: 前から・後ろからの並び(短点・長点の順番)を読む
  平均と最大でまとめて、65文字(音の63文字 + 括弧)のどれかを判定する
"""

from __future__ import annotations

import torch
from torch import nn

from light_features import INPUT_LENGTH


class MorseLightNet(nn.Module):
    def __init__(self, num_classes: int = 65, hidden: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),  # 144 → 72 点(1単位 = 3点)
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.gru = nn.GRU(64, hidden, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(4 * hidden, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)  # (件数, 64, 72)
        h, _ = self.gru(h.transpose(1, 2))  # (件数, 72, 2*hidden)
        pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=1)
        return self.classifier(pooled)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = MorseLightNet()
    print(count_parameters(m), m(torch.zeros(2, 1, INPUT_LENGTH)).shape)
