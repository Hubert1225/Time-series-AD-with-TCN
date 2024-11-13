"""This module provides classes of subsequence anomaly
detectors
"""

from abc import ABC, abstractmethod

import numpy as np


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
