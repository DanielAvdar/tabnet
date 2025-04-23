"""Log loss metric implementation."""

import torch
from torch.nn import CrossEntropyLoss

from .base_metrics import Metric


class LogLoss(Metric):
    """Log loss (cross-entropy) metric for classification tasks."""

    _name: str = "logloss"
    _maximize: bool = False

    def __call__(self, y_true: torch.Tensor, y_score: torch.Tensor, weights: torch.Tensor = None) -> float:
        """Compute the log loss (cross-entropy).

        Parameters
        ----------
        y_true : torch.Tensor
            True class labels.
        y_score : torch.Tensor
            Predicted scores or probabilities.
        weights : torch.Tensor, optional
            Sample weights.

        Returns
        -------
        float
            The computed log loss (cross-entropy).

        """
        loss = CrossEntropyLoss(reduction="none")(y_score.float(), y_true.long())
        if weights is not None:
            loss *= weights.to(y_true.device)
        return loss.mean().item()
