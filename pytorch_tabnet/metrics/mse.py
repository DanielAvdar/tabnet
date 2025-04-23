import torch

from .base_metrics import Metric


class MSE(Metric):
    _name: str = "mse"
    _maximize: bool = False

    def __call__(self, y_true: torch.Tensor, y_score: torch.Tensor, weights: torch.Tensor = None) -> float:
        errors = (y_true - y_score) ** 2
        if weights is not None:
            errors *= weights.to(y_true.device)
        return torch.mean(errors).cpu().item()
