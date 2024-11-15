"""This module provides classes of subsequence anomaly
detectors
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import Any

import numpy as np
import torch
from sklearn.neighbors import LocalOutlierFactor

from data_loading import TimeSeriesWithAnoms
from utils import sliding_window
from networks import TCNAutoencoder
from detection import (
    mahalanobis_anomaly_score,
    get_reconstruction_errors,
    get_k_max_nonoverlapping,
)
from params import TcnAeParams


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
    ) -> list[tuple[int, int]]:
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

    @staticmethod
    def load(
        series: TimeSeriesWithAnoms,
        params: TcnAeParams,
    ) -> TcnAeDetector:
        """Loads trained TCN autoencoder on a series
        and creates a detector for this series

        Args:
            series: series for which the model to load
            params: ``TcnAeParams`` object containing parameters values used
                to create TCN autoencoder object

        Returns:
            ``TcnAeDetector`` object

        """
        model = TCNAutoencoder(
            in_channels=(1 if series.values.ndim == 1 else series.values.shape[-2]),
            enc_channels=params.enc_channels,
            hidden_dim=params.hidden_dim,
            dec_channels=params.dec_channels,
            input_size=series.values.shape[-1],
            dilation_base=params.dilation_base,
            kernel_size=params.kernel_size,
        )
        model.load_state_dict(
            torch.load(
                os.path.join(params.checkpoints_dir, series.name + ".pth"),
                weights_only=True,
            )
        )
        model.eval()
        return TcnAeDetector(model)

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


##########################    BASELINE MODELS    ################################


class RandomDetector(SubsequenceAnomalyDetector):
    """Baseline subsequence anomaly detector.
    Detects random non-overlapping subsequences
    """

    name = "random"

    def detect(
        self,
        values: np.ndarray,
        anom_len: int,
        k_anoms: int,
    ) -> list[tuple[int, int]]:
        random_inds = np.random.permutation(values.shape[-1] - anom_len)
        anom_inds = get_k_max_nonoverlapping(random_inds, anom_len, k_anoms)
        return [(start_ind, start_ind + anom_len) for start_ind in anom_inds]


class LofDetector(SubsequenceAnomalyDetector):
    """Subsequence anomaly detector based on the Local Outlier Factor
    algorithm (LOF)

    During detection, all windows of length ``anom_len`` are extracted
    from the series. Then, each window is treated as an ``anom_len``-dimensional
    sample. All windows are passed to the LOF algorithm

    """

    def __init__(
        self,
        n_neighbors: int = 50,
        other_lof_params: dict[str, Any] | None = None,
    ):
        other_lof_params = {} if other_lof_params is None else other_lof_params
        self.lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            **other_lof_params,
        )

    def detect(
        self,
        values: np.ndarray,
        anom_len: int,
        k_anoms: int,
    ) -> list[tuple[int, int]]:
        series_windows = sliding_window(values, anom_len)
        self.lof.fit_predict(series_windows)
        inds_sorted = np.argsort(self.lof.negative_outlier_factor_).tolist()
        anom_inds = get_k_max_nonoverlapping(inds_sorted, anom_len, k_anoms)
        return [(start_ind, start_ind + anom_len) for start_ind in anom_inds]
