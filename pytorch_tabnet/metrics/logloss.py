import torch
from torch.nn import CrossEntropyLoss

from .base_metrics import Metric


class LogLoss(Metric):
    _name: str = "logloss"
    _maximize: bool = False

    def __call__(self, y_true: torch.Tensor, y_score: torch.Tensor, weights: torch.Tensor = None) -> float:
        loss = CrossEntropyLoss(reduction="none")(y_score.float(), y_true.long())
        if weights is not None:
            loss *= weights.to(y_true.device)
        return loss.mean().item()
