import torch

from .base_metrics import Metric


class MAE(Metric):
    _name: str = "mae"
    _maximize: bool = False

    def __call__(self, y_true: torch.Tensor, y_score: torch.Tensor, weights: torch.Tensor = None) -> float:
        errors = torch.abs(y_true - y_score)
        if weights is not None:
            errors *= weights.to(y_true.device)
        return torch.mean(errors).cpu().item()
