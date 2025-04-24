"""AUC metric implementation."""

import torch
from torcheval.metrics.functional import multiclass_auroc

from .base_metrics import Metric


class AUC(Metric):
    """Area Under the Curve (AUC) metric for classification tasks."""

    _name: str = "auc"
    _maximize: bool = True

    def __call__(self, y_true: torch.Tensor, y_score: torch.Tensor, weights: torch.Tensor = None) -> float:
        """Compute the AUC score.

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
            The computed AUC score.

        """
        num_of_classes = y_score.shape[1]
        return multiclass_auroc(y_score, y_true, num_classes=num_of_classes, average="macro").cpu().item()
