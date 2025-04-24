"""GBN class module."""

import math

import torch
from torch.nn import BatchNorm1d


class GBN(torch.nn.Module):
    """Ghost Batch Normalization.

    See: https://arxiv.org/abs/1705.08741.
    """

    def __init__(self, input_dim: int, virtual_batch_size: int = 128, momentum: float = 0.01):
        """Initialize Ghost Batch Normalization.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        virtual_batch_size : int
            Size of virtual batch for normalization.
        momentum : float
            Momentum for batch normalization.

        """
        super(GBN, self).__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Ghost Batch Normalization to input tensor x.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Normalized tensor.

        """
        v = x.shape[0] / self.virtual_batch_size
        v_ceil = math.ceil(v)
        chunks = x.chunk(v_ceil, dim=0)
        res = [self.bn(x_) for x_ in chunks]
        return torch.cat(res, dim=0)
