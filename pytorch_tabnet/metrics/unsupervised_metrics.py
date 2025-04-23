"""Unsupervised metrics for reconstruction tasks in PyTorch TabNet.

Provides metrics for evaluating unsupervised loss during model training and evaluation.
"""

import torch

from pytorch_tabnet.metrics import Metric, UnsupervisedLoss


class UnsupervisedMetric(Metric):
    """Unsupervised loss metric for reconstruction tasks."""

    _name: str = "unsup_loss"
    _maximize: bool = False

    def __call__(self, y_pred: torch.Tensor, embedded_x: torch.Tensor, obf_vars: torch.Tensor, weights: torch.Tensor = None) -> float:  # type: ignore[override]
        """Compute the unsupervised loss metric.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values.
        embedded_x : torch.Tensor
            Embedded input values.
        obf_vars : torch.Tensor
            Obfuscated variables mask.
        weights : torch.Tensor, optional
            Sample weights.

        Returns
        -------
        float
            The computed unsupervised loss metric.

        """
        loss = UnsupervisedLoss(y_pred, embedded_x, obf_vars, weights=weights)
        return loss.cpu().item()


class UnsupervisedNumpyMetric(Metric):
    """Unsupervised loss metric (NumPy version) for reconstruction tasks."""

    _name: str = "unsup_loss_numpy"
    _maximize: bool = False

    def __call__(self, y_pred: torch.Tensor, embedded_x: torch.Tensor, obf_vars: torch.Tensor, weights: torch.Tensor = None) -> float:  # type: ignore[override]
        """Compute the unsupervised loss metric (NumPy version).

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values.
        embedded_x : torch.Tensor
            Embedded input values.
        obf_vars : torch.Tensor
            Obfuscated variables mask.
        weights : torch.Tensor, optional
            Sample weights.

        Returns
        -------
        float
            The computed unsupervised loss metric (NumPy version).

        """
        return UnsupervisedLoss(y_pred, embedded_x, obf_vars).cpu().item()
