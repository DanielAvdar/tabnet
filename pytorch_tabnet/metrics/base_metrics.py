from dataclasses import dataclass
from typing import Any, List, Union

import torch


def UnsupervisedLoss(
    y_pred: torch.Tensor,
    embedded_x: torch.Tensor,
    obf_vars: torch.Tensor,
    eps: float = 1e-9,
    weights: torch.Tensor = None,
) -> torch.Tensor:
    errors = y_pred - embedded_x
    reconstruction_errors = torch.mul(errors, obf_vars) ** 2
    batch_means = torch.mean(embedded_x, dim=0)
    batch_means[batch_means == 0] = 1
    batch_stds = torch.std(embedded_x, dim=0) ** 2
    batch_stds[batch_stds == 0] = batch_means[batch_stds == 0]
    features_loss = torch.matmul(reconstruction_errors, 1 / batch_stds)
    nb_reconstructed_variables = torch.sum(obf_vars, dim=1)
    features_loss = features_loss / (nb_reconstructed_variables + eps)
    if weights is not None:
        features_loss = features_loss * weights
    loss = torch.mean(features_loss)
    return loss


class Metric:
    """Abstract base class for defining custom metrics."""

    _name: str
    _maximize: bool

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor, weights: torch.Tensor = None) -> float:
        raise NotImplementedError("Custom Metrics must implement this function")

    @classmethod
    def get_metrics_by_names(cls, names: List[str]) -> List:
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
    metric_names: List[str]
    prefix: str = ""

    def __post_init__(self) -> None:
        self.metrics = Metric.get_metrics_by_names(self.metric_names)
        self.names = [self.prefix + name for name in self.metric_names]

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor, weights: torch.Tensor = None) -> dict:
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
    metric_names: List[str]
    prefix: str = ""

    def __post_init__(self) -> None:
        self.metrics = Metric.get_metrics_by_names(self.metric_names)
        self.names = [self.prefix + name for name in self.metric_names]

    def __call__(self, y_pred: torch.Tensor, embedded_x: torch.Tensor, obf_vars: torch.Tensor, weights: torch.Tensor = None) -> dict:
        logs = {}
        for metric in self.metrics:
            res = metric(y_pred, embedded_x, obf_vars, weights)
            logs[self.prefix + metric._name] = res
        return logs


def check_metrics(metrics: List[Union[str, Any]]) -> List[str]:
    val_metrics = []
    for metric in metrics:
        if isinstance(metric, str):
            val_metrics.append(metric)
        elif issubclass(metric, Metric):
            val_metrics.append(metric()._name)
        else:
            raise TypeError("You need to provide a valid metric format")
    return val_metrics


class UnsupervisedMetric(Metric):
    _name: str = "unsup_loss"
    _maximize: bool = False

    def __call__(self, y_pred: torch.Tensor, embedded_x: torch.Tensor, obf_vars: torch.Tensor, weights: torch.Tensor = None) -> float:
        loss = UnsupervisedLoss(y_pred, embedded_x, obf_vars, weights=weights)
        return loss.cpu().item()


class UnsupervisedNumpyMetric(Metric):
    _name: str = "unsup_loss_numpy"
    _maximize: bool = False

    def __call__(self, y_pred: torch.Tensor, embedded_x: torch.Tensor, obf_vars: torch.Tensor, weights: torch.Tensor = None) -> float:
        return UnsupervisedLoss(y_pred, embedded_x, obf_vars).cpu().item()
