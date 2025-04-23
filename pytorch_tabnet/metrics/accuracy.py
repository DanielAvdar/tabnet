import torch
from torcheval.metrics.functional import multiclass_accuracy

from .base_metrics import Metric


class Accuracy(Metric):
    _name: str = "accuracy"
    _maximize: bool = True

    def __call__(self, y_true: torch.Tensor, y_score: torch.Tensor, weights: torch.Tensor = None) -> float:
        res = multiclass_accuracy(y_score, y_true)
        return res.cpu().item()
