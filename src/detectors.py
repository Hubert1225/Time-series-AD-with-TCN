"""This module provides classes of subsequence anomaly
detectors
"""

from abc import ABC, abstractmethod

import numpy as np

from utils import sliding_window
from networks import TCNAutoencoder
from detection import (
    mahalanobis_anomaly_score,
    get_reconstruction_errors,
    get_k_max_nonoverlapping,
)


class SubsequenceAnomalyDetector(ABC):
    """Base class for all time series subsequence
    anomaly detectors
    """

    name: str

    @abstractmethod
    def detect(
        self,
        values: np.ndarray,
        anom_len: int,
        k_anoms: int,
    ) -> list[int, int]:
        """Detects subsequence anomalies in a time series
        values

        Args:
            values: input time series - array of shape (input_size,)
            model: trained TCN autoencoder
            anom_len: anomaly length
            k_anoms: number of anomalies to detect

        Returns:
            list of tuples, each element is (start_index, end_index + 1) for an anomaly

        """
        pass


class TcnAeDetector(SubsequenceAnomalyDetector):
    """Subsequence anomaly detector that detect anomalies
    using the reconstruction error of a TCN autoencoder

    Args:
        model: trained TCN autoencoder

    """

    name = "TCN Autoencoder"

    def __init__(self, model: TCNAutoencoder):
        self._model = model

    def detect(
        self,
        values: np.ndarray,
        anom_len: int,
        k_anoms: int,
    ) -> list[tuple[int, int]]:
        recon_errors = get_reconstruction_errors(values, self._model)
        errors_sliding_windows = sliding_window(recon_errors, anom_len)
        windows_anomaly_score = mahalanobis_anomaly_score(errors_sliding_windows)
        inds_sorted = np.flip(np.argsort(windows_anomaly_score)).tolist()
        anom_inds = get_k_max_nonoverlapping(inds_sorted, anom_len, k_anoms)
        return [(start_ind, start_ind + anom_len) for start_ind in anom_inds]
