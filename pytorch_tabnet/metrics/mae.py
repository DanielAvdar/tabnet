"""Mean Absolute Error (MAE) metric implementation."""

import torch

from .base_metrics import Metric


class MAE(Metric):
    """Mean Absolute Error (MAE) metric for regression tasks."""

    _name: str = "mae"
    _maximize: bool = False

    def __call__(self, y_true: torch.Tensor, y_score: torch.Tensor, weights: torch.Tensor = None) -> float:
        """Compute the mean absolute error (MAE).

        Parameters
        ----------
        y_true : torch.Tensor
            True values.
        y_score : torch.Tensor
            Predicted values.
        weights : torch.Tensor, optional
            Sample weights.

        Returns
        -------
        float
            The computed mean absolute error (MAE).

        """
        errors = torch.abs(y_true - y_score)
        if weights is not None:
            errors *= weights.to(y_true.device)
        return torch.mean(errors).cpu().item()
