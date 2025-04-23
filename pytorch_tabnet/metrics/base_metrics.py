"""Base metrics and metric containers for pytorch_tabnet."""

from dataclasses import dataclass
from typing import Any, List, Union

import torch


class Metric:
    """Abstract base class for defining custom metrics."""

    _name: str
    _maximize: bool

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor, weights: torch.Tensor = None) -> float:
        """Compute the metric value. Must be implemented by subclasses."""
        raise NotImplementedError("Custom Metrics must implement this function")

    @classmethod
    def get_metrics_by_names(cls, names: List[str]) -> List:
        """Get metric instances by their names.

        Parameters
        ----------
        names : list of str
            List of metric names.

        Returns
        -------
        list
            List of metric instances.

        """
        available_metrics = cls.__subclasses__()
        available_names = [metric()._name for metric in available_metrics]
        metrics = []
        for name in names:
            assert name in available_names, f"{name} is not available, choose in {available_names}"
            idx = available_names.index(name)
            metric = available_metrics[idx]()
            metrics.append(metric)
        return metrics


@dataclass
class MetricContainer:
    """Container for managing multiple supervised metrics."""

    metric_names: List[str]
    prefix: str = ""

    def __post_init__(self) -> None:
        """Initialize the metric container."""
        self.metrics = Metric.get_metrics_by_names(self.metric_names)
        self.names = [self.prefix + name for name in self.metric_names]

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor, weights: torch.Tensor = None) -> dict:
        """Compute all metrics in the container.

        Parameters
        ----------
        y_true : torch.Tensor
            True values.
        y_pred : torch.Tensor
            Predicted values.
        weights : torch.Tensor, optional
            Sample weights.

        Returns
        -------
        dict
            Dictionary of metric names and their computed values.

        """
        logs = {}
        for metric in self.metrics:
            if isinstance(y_pred, list):
                res = torch.mean(torch.tensor([metric(y_true[:, i], y_pred[i], weights) for i in range(len(y_pred))]))
            else:
                res = metric(y_true, y_pred, weights)
            logs[self.prefix + metric._name] = res
        return logs


@dataclass
class UnsupMetricContainer:
    """Container for managing multiple unsupervised metrics."""

    metric_names: List[str]
    prefix: str = ""

    def __post_init__(self) -> None:
        """Initialize the unsupervised metric container."""
        self.metrics = Metric.get_metrics_by_names(self.metric_names)
        self.names = [self.prefix + name for name in self.metric_names]

    def __call__(self, y_pred: torch.Tensor, embedded_x: torch.Tensor, obf_vars: torch.Tensor, weights: torch.Tensor = None) -> dict:
        """Compute all unsupervised metrics in the container.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values.
        embedded_x : torch.Tensor
            Embedded input values.
        obf_vars : torch.Tensor
            Obfuscated variables mask.
        weights : torch.Tensor, optional
            Sample weights.

        Returns
        -------
        dict
            Dictionary of unsupervised metric names and their computed values.

        """
        logs = {}
        for metric in self.metrics:
            res = metric(y_pred, embedded_x, obf_vars, weights)
            logs[self.prefix + metric._name] = res
        return logs


def check_metrics(metrics: List[Union[str, Any]]) -> List[str]:
    """Validate and return metric names from a list of metrics or strings.

    Parameters
    ----------
    metrics : list of str or Metric
        List of metric names or Metric classes.

    Returns
    -------
    list of str
        List of valid metric names.

    Raises
    ------
    TypeError
        If a metric is not a string or Metric subclass.

    """
    val_metrics = []
    for metric in metrics:
        if isinstance(metric, str):
            val_metrics.append(metric)
        elif issubclass(metric, Metric):
            val_metrics.append(metric()._name)
        else:
            raise TypeError("You need to provide a valid metric format")
    return val_metrics
