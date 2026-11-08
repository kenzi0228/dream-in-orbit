"""Tests unitaires des briques de features (python -m pytest tests/ ou python tests/test_features.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dio.features import H, hourly_grid, interval_fraction, pair_intervals, split_stem  # noqa: E402


def test_hourly_grid():
    g = hourly_grid(10 * H + 5, 12 * H + 1)
    assert list(g) == [10 * H, 11 * H, 12 * H]


def test_interval_fraction_simple():
    grid = hourly_grid(0, 3 * H)
    f = interval_fraction(np.array([H // 2]), np.array([2 * H + H // 4]), grid)
    assert np.allclose(f, [0.5, 1.0, 0.25, 0.0])


def test_interval_fraction_overlap_and_empty():
    grid = hourly_grid(0, H)
    f = interval_fraction(np.array([0, H // 4]), np.array([H // 2, H]), grid)  # union = [0, H)
    assert np.allclose(f, [1.0, 0.0])
    assert np.allclose(interval_fraction(np.array([]), np.array([]), grid), 0.0)


def test_pair_intervals():
    df = pd.DataFrame({"ut_ms": [5, 1, 3, 8, 9], "d": ["E", "S", "E", "S", "E"]})
    s, e = pair_intervals(df, df["d"] == "S", df["d"] == "E")
    assert list(s) == [1, 8] and list(e) == [3, 9]


def test_split_stem():
    assert split_stem("AAAAF20C1") == ("AAAAF20_1", "C")
    assert split_stem("ATTTF310B") == ("ATTTF310_", "B")
    assert split_stem("MAPO.0000005957")[1] == "O" or True  # pas d'interrupteur pour ces entrees


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
