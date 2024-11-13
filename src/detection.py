"""This module provides functions to perform anomaly detection
on a time series using a train autoencoder
"""
import numpy as np
from scipy.spatial.distance import mahalanobis

from utils import SkipIteration


def mahalanobis_anomaly_score(X: np.ndarray) -> np.ndarray:
    """Anomaly scores for samples based on their
    Mahalanobis distance

    Args:
        X: 2D array of samples

    Returns:
        1D array with anomaly scores for samples in X

    """
    mu = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False, bias=True)
    cov_inv = None
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        print("Singular covariance matrix - using regularization")
        n, _ = cov.shape
        i = 0
        while cov_inv is None:
            try:
                cov_inv = np.linalg.inv(cov + np.identity(n) * 0.01 * np.power(2, i))
            except np.linalg.LinAlgError:
                i += 1
    return np.array([mahalanobis(x, mu, cov_inv) for x in X])


def get_k_max_nonoverlapping(
    inds_sorted: list[int],
    window_len: int,
    k: int,
) -> list[int]:
    """Extracts k non-overlapping windows with the lowest or highest value
    of some property (e.g. anomaly score)

    Args:
        inds_sorted: list of start indices of windows sorted by the property
            (windows that occur earlier in the list have more probability to occur
            in the output)
        window_len: window length (number of observations in window)
        k: number of non-overlapping windows to extract

    Returns:
        k start indices of non-overlapping windows

    """
    k_max_inds = []

    for window_ind in inds_sorted:

        try:

            # check if window does not overlap with any windows in list
            for ind_in_list in k_max_inds:
                if np.abs(window_ind - ind_in_list) < window_len:
                    raise SkipIteration

            # no overlaps - add ind to the list
            k_max_inds.append(window_ind)

            # if k items in list - break
            if len(k_max_inds) == k:
                break

        except SkipIteration:
            pass

    return k_max_inds
