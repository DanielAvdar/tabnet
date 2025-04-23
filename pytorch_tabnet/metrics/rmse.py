import torch

from .base_metrics import Metric


class RMSE(Metric):
    _name: str = "rmse"
    _maximize: bool = False

    def __call__(self, y_true: torch.Tensor, y_score: torch.Tensor, weights: torch.Tensor = None) -> float:
        mse_errors = (y_true - y_score) ** 2
        if weights is not None:
            mse_errors *= weights.to(y_true.device)
        return torch.sqrt(torch.mean(mse_errors)).cpu().item()
