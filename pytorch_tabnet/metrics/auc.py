import torch
from torcheval.metrics.functional import multiclass_auroc

from .base_metrics import Metric


class AUC(Metric):
    _name: str = "auc"
    _maximize: bool = True

    def __call__(self, y_true: torch.Tensor, y_score: torch.Tensor, weights: torch.Tensor = None) -> float:
        num_of_classes = y_score.shape[1]
        return multiclass_auroc(y_score, y_true, num_classes=num_of_classes, average="macro").cpu().item()
