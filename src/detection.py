"""This module provides functions to perform anomaly detection
on a time series using a train autoencoder
"""

import numpy as np
from scipy.spatial.distance import mahalanobis
import torch

from utils import SkipIteration, sliding_window
from networks import TCNAutoencoder


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


def get_reconstruction_errors(
    values: np.ndarray,
    model: TCNAutoencoder,
) -> np.ndarray:
    """Obtains reconstruction errors (elementwise)
    of a given autoencoder on the given data

    Args:
        values: array of input data of shape (input_size,) or (n_channels, input_size)
        model: trained TCNAutoencoder model

    Returns:
        array of elementwise reconstruction errors of the autoencoder
        (of the same shape as the input values)

    Notes:
        the function does NOT take the absolute values of the errors, so
        each value is from the interval (-inf, inf)

    """
    model.eval()
    model.decoder.set_output_size(values.shape[-1])

    values_tensor = torch.tensor(values).float()
    if values_tensor.ndim == 1:
        values_tensor = values_tensor.unsqueeze(0)  # set number of channels to 1
    values_tensor = values_tensor.unsqueeze(0)  # set number of batches to 1

    values_recon = model(values_tensor)
    return (values_recon - values_tensor).squeeze().detach().numpy()


def detect_subsequence_anomalies(
    values: np.ndarray,
    model: TCNAutoencoder,
    anom_len: int,
    k_anoms: int,
) -> list[tuple[int, int]]:
    """Detects subsequence anomalies in time series values
    using a trained TCN Autoencoder

    Args:
        values: input time series - array of shape (input_size,)
        model: trained TCN autoencoder
        anom_len: anomaly length
        k_anoms: number of anomalies to detect

    Returns:
        list of tuples, each element is (start_index, end_index + 1) for an anomaly

    """
    recon_errors = get_reconstruction_errors(values, model)
    errors_sliding_windows = sliding_window(recon_errors, anom_len)
    windows_anomaly_score = mahalanobis_anomaly_score(errors_sliding_windows)
    inds_sorted = np.flip(np.argsort(windows_anomaly_score)).tolist()
    anom_inds = get_k_max_nonoverlapping(inds_sorted, anom_len, k_anoms)
    return [(start_ind, start_ind + anom_len) for start_ind in anom_inds]
