"""Accuracy metric implementation."""

import torch
from torcheval.metrics.functional import multiclass_accuracy

from .base_metrics import Metric


class Accuracy(Metric):
    """Accuracy metric for classification tasks."""

    _name: str = "accuracy"
    _maximize: bool = True

    def __call__(self, y_true: torch.Tensor, y_score: torch.Tensor, weights: torch.Tensor = None) -> float:
        """Compute the accuracy score.

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
            The computed accuracy score.

        """
        res = multiclass_accuracy(y_score, y_true)
        return res.cpu().item()
