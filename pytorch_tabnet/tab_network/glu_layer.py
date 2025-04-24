"""GLU_Layer class module."""

from typing import Optional

import torch
from torch.nn import Linear

from .gbn import GBN

# from pytorch_tabnet.tab_network import initialize_glu, GBN
from .utils_funcs import initialize_glu


class GLU_Layer(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        fc: Optional[Linear] = None,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
    ):
        """Initialize a GLU (Gated Linear Unit) layer.

        Parameters
        ----------
        input_dim : int
            Input dimension.
        output_dim : int
            Output dimension.
        fc : Optional[Linear]
            Optional linear layer to use.
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization.
        momentum : float
            BatchNorm momentum.

        """
        super(GLU_Layer, self).__init__()

        self.output_dim = output_dim
        if fc:
            self.fc = fc
        else:
            self.fc = Linear(input_dim, 2 * output_dim, bias=False)
        initialize_glu(self.fc, input_dim, 2 * output_dim)

        self.bn = GBN(2 * output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for GLU_Layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after GLU transformation.

        """
        x = self.fc(x)
        x = self.bn(x)
        out = torch.mul(x[:, : self.output_dim], torch.sigmoid(x[:, self.output_dim :]))
        return out
