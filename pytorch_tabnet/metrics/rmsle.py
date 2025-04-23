"""Root Mean Squared Logarithmic Error (RMSLE) metric implementation."""

import torch

from .base_metrics import Metric


class RMSLE(Metric):
    """Root Mean Squared Logarithmic Error (RMSLE) metric for regression tasks."""

    _name: str = "rmsle"
    _maximize: bool = False

    def __call__(self, y_true: torch.Tensor, y_score: torch.Tensor, weights: torch.Tensor = None) -> float:
        """Compute the root mean squared logarithmic error (RMSLE).

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
            The computed root mean squared logarithmic error (RMSLE).

        """
        logerror = torch.log(y_score + 1) - torch.log(y_true + 1)
        squared_logerror = logerror**2
        if weights is not None:
            squared_logerror *= weights.to(y_true.device)
        return torch.sqrt(torch.mean(squared_logerror)).cpu().item()
