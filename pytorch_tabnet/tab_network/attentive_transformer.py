"""AttentiveTransformer class module."""

from typing import Optional

import torch
from activations_plus import Entmax, Sparsemax
from torch.nn import Linear

from .gbn import GBN

# from pytorch_tabnet.tab_network import initialize_non_glu, GBN
from .utils_funcs import initialize_non_glu


class AttentiveTransformer(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        group_dim: int,
        group_matrix: Optional[torch.Tensor],
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
        mask_type: str = "sparsemax",
    ):
        """Initialize an attention transformer.

        Parameters
        ----------
        input_dim : int
            Input size
        group_dim : int
            Number of groups for features
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use

        """
        super(AttentiveTransformer, self).__init__()
        self.fc = Linear(input_dim, group_dim, bias=False)
        initialize_non_glu(self.fc, input_dim, group_dim)
        self.bn = GBN(group_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)

        if mask_type == "sparsemax":
            # Sparsemax
            self.selector = Sparsemax(dim=-1)
        elif mask_type == "entmax":
            # Entmax
            self.selector = Entmax(dim=-1)
        else:
            raise NotImplementedError("Please choose either sparsemax" + "or entmax as masktype")

    def forward(self, priors: torch.Tensor, processed_feat: torch.Tensor) -> torch.Tensor:
        x = self.fc(processed_feat)
        x = self.bn(x)
        x = torch.mul(x, priors)
        x = self.selector(x)
        return x
