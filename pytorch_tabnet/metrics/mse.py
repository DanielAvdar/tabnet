"""Mean Squared Error (MSE) metric implementation."""

import torch

from .base_metrics import Metric


class MSE(Metric):
    """Mean Squared Error (MSE) metric for regression tasks."""

    _name: str = "mse"
    _maximize: bool = False

    def __call__(self, y_true: torch.Tensor, y_score: torch.Tensor, weights: torch.Tensor = None) -> float:
        """Compute the mean squared error (MSE).

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
            The computed mean squared error (MSE).

        """
        errors = (y_true - y_score) ** 2
        if weights is not None:
            errors *= weights.to(y_true.device)
        return torch.mean(errors).cpu().item()
