"""Root Mean Squared Error (RMSE) metric implementation."""

import torch

from .base_metrics import Metric


class RMSE(Metric):
    """Root Mean Squared Error (RMSE) metric for regression tasks."""

    _name: str = "rmse"
    _maximize: bool = False

    def __call__(self, y_true: torch.Tensor, y_score: torch.Tensor, weights: torch.Tensor = None) -> float:
        """Compute the root mean squared error (RMSE).

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
            The computed root mean squared error (RMSE).

        """
        mse_errors = (y_true - y_score) ** 2
        if weights is not None:
            mse_errors *= weights.to(y_true.device)
        return torch.sqrt(torch.mean(mse_errors)).cpu().item()
