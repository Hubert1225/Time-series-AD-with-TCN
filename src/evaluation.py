from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import f1_score

from data_loading import TimeSeriesWithAnoms


class EvaluationMetric(ABC):
    """Base class for all metrics for evaluating
    results for subsequence anomaly detection
    """

    name: str

    @abstractmethod
    def evaluate(self, ts: TimeSeriesWithAnoms, detected: list[tuple[int, int]]) -> float:
        """
        Args:
            ts: time series object for which anomaly detection has been performed
                (containing ground truth annotations)
            detected: list of tuples: (start_index, end_index + 1), where
                each tuple represents one detected anomaly

        Returns:
            metric value
        """
        pass


def get_labels_series(length: int, ones_periods: list[tuple[int, int]]) -> np.ndarray:
    """Returns binary series with ones
    in desired periods
    """
    series = np.zeros(length, dtype=np.uint8)
    for period_start, period_end in ones_periods:
        series[period_start:period_end] = 1
    return series


class F1Score(EvaluationMetric):

    name = 'f1-score'

    def evaluate(self, ts: TimeSeriesWithAnoms, detected: list[tuple[int, int]]) -> float:
        ground_truth_labels = get_labels_series(ts.values.shape[0], ts.annotations)
        detected_labels = get_labels_series(ts.values.shape[0], detected)
        return f1_score(
            y_true=ground_truth_labels,
            y_pred=detected_labels,
        )


class PrecisionAtK(EvaluationMetric):

    name = "precision@k"

    @staticmethod
    def is_detection_relevant(
        ground_truth_annotations: list[tuple[int, int]],
        detected_anomaly: tuple[int, int],
        series_len: int,
    ) -> bool:
        ground_truth_labels = get_labels_series(series_len, ground_truth_annotations)
        detected_labels = get_labels_series(series_len, [detected_anomaly])
        return np.sum(ground_truth_labels & detected_labels) > 0

    def evaluate(self, ts: TimeSeriesWithAnoms, detected: list[tuple[int, int]]) -> float:
        return float(np.mean(
            list(map(lambda x: self.is_detection_relevant(ts.annotations, x, ts.values.shape[0]), detected))
        ))