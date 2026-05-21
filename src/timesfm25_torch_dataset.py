"""PyTorch Dataset classes for TimesFM 2.5 fine-tuning.

Provides:
    load_series_from_dir()             -- scan a directory of weekly CSVs, return list[np.ndarray]
    TimeSeriesRandomWindowDataset      -- random-window sampling for training
    TimeSeriesLastWindowDataset        -- last-window eval dataset

No external normalisation is applied -- TimesFM 2.5 handles RevIN internally.
Items are returned as plain numpy arrays so the module can be imported and tested
without torch being installed.  The DataLoader will call torch.utils.data.default_collate
which converts numpy float32 arrays to tensors automatically.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from torch.utils.data import Dataset as _TorchDataset
    _BASE = _TorchDataset
except ImportError:  # torch not installed (unit-test environment)
    class _TorchDataset:  # type: ignore
        pass
    _BASE = _TorchDataset


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_series_from_dir(
    data_dir: str | Path,
    target_col: str = "log_return_1",
    dropna: bool = True,
    glob: str = "*.csv",
) -> list[np.ndarray]:
    """Read all CSV files in data_dir and return one float32 array per file.

    Args:
        data_dir:   Directory containing weekly EURUSD CSV files.
        target_col: Column to extract as the time series.
        dropna:     If True, drop NaN rows before extracting the series.
        glob:       File glob pattern (default '*.csv').

    Returns:
        List of 1-D float32 numpy arrays, one per CSV file, sorted by filename.

    Raises:
        FileNotFoundError: if data_dir does not exist.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    series_list: list[np.ndarray] = []
    for csv_path in sorted(data_dir.glob(glob)):
        df = pd.read_csv(csv_path)
        if target_col not in df.columns:
            continue
        if dropna:
            df = df.dropna(subset=[target_col])
        values = df[target_col].values.astype(np.float32)
        series_list.append(values)

    return series_list


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class TimeSeriesRandomWindowDataset(_BASE):
    """Random-window dataset for TimesFM 2.5 fine-tuning.

    Pre-samples num_samples random (series_idx, start_pos) pairs at
    construction time.  Each __getitem__ slices a contiguous
    [context_len + horizon_len] window and returns:
        context : float32 ndarray [context_len]
        target  : float32 ndarray [horizon_len]

    No external normalisation -- TimesFM handles instance normalisation
    (RevIN) internally.

    Args:
        series_list: List of 1-D float32 arrays.
        context_len: Number of history steps fed to the model.
                     Must be a multiple of 32 (TimesFM patch size).
        horizon_len: Number of future steps to forecast.
        num_samples: Number of random windows to pre-sample.
        seed:        RNG seed for reproducibility.
    """

    def __init__(
        self,
        series_list: list[np.ndarray],
        context_len: int,
        horizon_len: int,
        num_samples: int = 5000,
        seed: int = 42,
    ) -> None:
        self.series_list = series_list
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.samples: list[tuple[int, int]] = []

        rng = np.random.default_rng(seed)
        min_len = context_len + horizon_len
        valid_indices = [i for i, s in enumerate(series_list) if len(s) >= min_len]

        if not valid_indices:
            shortest = min((len(s) for s in series_list), default=0)
            raise ValueError(
                f"No series long enough for context_len={context_len} + "
                f"horizon_len={horizon_len} (need >= {min_len}). "
                f"Shortest series has {shortest} steps."
            )

        for _ in range(num_samples):
            idx = int(rng.choice(valid_indices))
            series = series_list[idx]
            max_start = len(series) - min_len
            start = int(rng.integers(0, max_start + 1))
            self.samples.append((idx, start))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        idx, start = self.samples[i]
        series = self.series_list[idx]
        context = series[start : start + self.context_len].astype(np.float32)
        target = series[start + self.context_len : start + self.context_len + self.horizon_len].astype(np.float32)
        return context, target


class TimeSeriesLastWindowDataset(_BASE):
    """Validation dataset using the final window of each series.

    Each item is the last [context_len + horizon_len] slice of a series:
        context : float32 ndarray [context_len]    -- the last context_len steps
        target  : float32 ndarray [horizon_len]    -- the last horizon_len steps

    Series shorter than context_len + horizon_len are silently excluded.

    Args:
        series_list: List of 1-D float32 arrays.
        context_len: Number of history steps.
        horizon_len: Number of future steps.
    """

    def __init__(
        self,
        series_list: list[np.ndarray],
        context_len: int,
        horizon_len: int,
    ) -> None:
        self.context_len = context_len
        self.horizon_len = horizon_len
        min_len = context_len + horizon_len
        self.items: list[tuple[np.ndarray, np.ndarray]] = []
        for s in series_list:
            if len(s) >= min_len:
                ctx = s[-min_len : -horizon_len].astype(np.float32)
                tgt = s[-horizon_len:].astype(np.float32)
                self.items.append((ctx, tgt))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        return self.items[i]
