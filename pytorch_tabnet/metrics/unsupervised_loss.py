"""Unsupervised loss computation for reconstruction tasks in PyTorch TabNet.

Defines the loss function used to evaluate reconstruction performance in unsupervised settings.
"""

import torch


def UnsupervisedLoss(
    y_pred: torch.Tensor,
    embedded_x: torch.Tensor,
    obf_vars: torch.Tensor,
    eps: float = 1e-9,
    weights: torch.Tensor = None,
) -> torch.Tensor:
    """Compute the unsupervised loss for reconstruction tasks.

    Parameters
    ----------
    y_pred : torch.Tensor
        Predicted values.
    embedded_x : torch.Tensor
        Embedded input values.
    obf_vars : torch.Tensor
        Obfuscated variables mask.
    eps : float, optional
        Small value to avoid division by zero.
    weights : torch.Tensor, optional
        Sample weights.

    Returns
    -------
    torch.Tensor
        The computed unsupervised reconstruction loss.

    """
    errors = y_pred - embedded_x
    reconstruction_errors = torch.mul(errors, obf_vars) ** 2
    batch_means = torch.mean(embedded_x, dim=0)
    batch_means[batch_means == 0] = 1
    batch_stds = torch.std(embedded_x, dim=0) ** 2
    batch_stds[batch_stds == 0] = batch_means[batch_stds == 0]
    features_loss = torch.matmul(reconstruction_errors, 1 / batch_stds)
    nb_reconstructed_variables = torch.sum(obf_vars, dim=1)
    features_loss = features_loss / (nb_reconstructed_variables + eps)
    if weights is not None:
        features_loss = features_loss * weights
    loss = torch.mean(features_loss)
    return loss
