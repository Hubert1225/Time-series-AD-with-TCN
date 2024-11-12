"""This module provides utils for convenient
access to the benchmark dataset
"""

from dataclasses import dataclass
import os
import re
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import nonoverlap_sliding_windows

srw_regex = re.compile(
    r"^SinusRW_Length_(\d+)_AnomalyL_(\d+)_AnomalyN_(\d+)_NoisePerc_\d+$"
)


@dataclass
class TimeSeriesWithAnoms:
    """Represents one time series. Stores
    time series values and anomalous subsequences
    annotations

    Args:
        name: string used as a series identifier
        values: 1D array with time series values
        annotations: list of tuples, where each tuple: (start_index, last_index + 1)
            represents one subsequence anomaly

    """

    name: str
    values: np.ndarray
    annotations: list[tuple[int, int]]


def get_srw_series(dir_path: str, series_name: str) -> TimeSeriesWithAnoms:
    """Loads a time series from SyntheticRandomWalk dataset

    Args:
        dir_path: path to the directory with data and annotation files
        series_name: name of the series file without
            extension, e.g. `SinusRW_Length_120000_AnomalyL_200_AnomalyN_100_NoisePerc_0`

    Returns:
        TimeSeriesWithAnoms object

    """
    series_metadata_vals = srw_regex.findall(series_name)[0]
    series_length, anom_length, n_anoms = [int(val) for val in series_metadata_vals]

    series_path = os.path.join(dir_path, series_name + ".ts")
    ann_path = os.path.join(dir_path, series_name + "_Annotations.txt")

    with open(ann_path) as f:
        ann_inds = [int(line.strip()) for line in f.readlines()]
    anoms = [(ind, ind + anom_length) for ind in ann_inds]

    values = pd.read_csv(series_path, header=None).values.reshape(-1)

    return TimeSeriesWithAnoms(name=series_name, values=values, annotations=anoms)


def load_all_srw_series(
    dir_path: str = "data/srw",
) -> list[TimeSeriesWithAnoms]:
    """Loads all series from SinusRandomWalk dataset saved
    in a given directory
    """
    all_names = [os.path.splitext(fname)[0] for fname in os.listdir(dir_path)]
    series_names = [name for name in all_names if not name.endswith("Annotations")]
    return [get_srw_series(dir_path, series_name) for series_name in series_names]


class SlidingWindowDataset(Dataset):

    def __init__(self, ts_values: np.ndarray, window_len: int):
        self.windows = nonoverlap_sliding_windows(ts_values, window_len)

    def __len__(self):
        return self.windows.shape[0]

    def __getitem__(self, idx):
        window = self.windows[idx, :]
        return torch.tensor(window, dtype=torch.float32).reshape((1, -1))
