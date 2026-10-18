import csv
from collections import Counter

import numpy as np

from generate_light_dataset import assign_splits, generate, load_traces
from light_table import CLASS_CODES


def test_assign_splits_ratio():
    splits = assign_splits(300, np.random.default_rng(0))
    assert Counter(splits) == {"train": 210, "val": 45, "test": 45}


def test_generate_small_dataset(tmp_path):
    meta = generate(tmp_path, per_class=4, seed=1)
    with open(meta, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    traces = load_traces(tmp_path)
    assert len(rows) == len(traces) == 65 * 4
    assert {r["label"] for r in rows} == set(CLASS_CODES)
    assert all(int(r["frames"]) == len(t) for r, t in zip(rows, traces))
    assert all(0 <= t.min() and t.max() <= 255 for t in traces)


def test_generate_is_reproducible(tmp_path):
    generate(tmp_path / "a", per_class=2, seed=3)
    generate(tmp_path / "b", per_class=2, seed=3)
    a, b = load_traces(tmp_path / "a"), load_traces(tmp_path / "b")
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
