from abc import ABC, abstractmethod

from data_loading import TimeSeriesWithAnoms


class EvaluationMetric(ABC):
    """Base class for all metrics for evaluating
    results for subsequence anomaly detection
    """

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
