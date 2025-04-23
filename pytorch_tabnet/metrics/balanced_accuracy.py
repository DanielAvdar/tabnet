import torch
from torcheval.metrics.functional import multiclass_accuracy

from .base_metrics import Metric


class BalancedAccuracy(Metric):
    _name: str = "balanced_accuracy"
    _maximize: bool = True

    def __call__(self, y_true: torch.Tensor, y_score: torch.Tensor, weights: torch.Tensor = None) -> float:
        num_of_classes = y_score.shape[1]
        return multiclass_accuracy(y_score, y_true, average="macro", num_classes=num_of_classes).cpu().item()
