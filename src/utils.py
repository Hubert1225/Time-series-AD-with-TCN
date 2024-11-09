import numpy as np


def sliding_window(ts: np.ndarray, window_len: int) -> np.ndarray:
    """Extracts subsequent sliding windows
    of given series values

    Args:
        ts (1D array): series values
        window_len (int): length of the sliding window

    Returns:
        2D array: array with subsequent sliding windows,
            of shape (n_windows, window_len),
            where n_windows is equal to ts_len - window_len + 1
    """
    n_windows = ts.shape[0] - window_len + 1

    sliding_inds = np.tile(np.arange(window_len), (n_windows, 1)) + np.arange(
        n_windows
    ).reshape((-1, 1))

    return ts[sliding_inds]


def nonoverlap_sliding_windows(ts: np.ndarray, window_len: int) -> np.ndarray:
    """Extract subsequent sliding windows with no overlap

    Args:
        ts (1D array): series values
        window_len (int): length of the sliding window

    Returns:
        2D array: array with subsequent sliding windows,
            of shape (n_windows, window_len),
            where n_windows is equal to floor(ts_len / window_len)
    """
    n_windows = ts.shape[0] // window_len
    sliding_inds = np.tile(np.arange(window_len), (n_windows, 1)) + np.arange(
        0, n_windows * window_len, window_len
    ).reshape((-1, 1))
    return ts[sliding_inds]


def normalize_values(values: np.ndarray) -> np.ndarray:
    """Normalizes all values in a given array
    to the interval [0.0; 1.0]

    If all values are the same, then all values are set
    to 1.0
    """
    values_min = np.amin(values)
    values_max = np.amax(values)
    return (
        np.ones_like(values)
        if values_min == values_max
        else (values - values_min) / (values_max - values_min)
    )
