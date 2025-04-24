"""FeatTransformer class module."""

from typing import Optional

import torch

# from pytorch_tabnet.tab_network.glu_block import GLU_Block
from .glu_block import GLU_Block


class FeatTransformer(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        shared_layers: Optional[torch.nn.ModuleList],
        n_glu_independent: int,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
    ):
        super(FeatTransformer, self).__init__()
        """
        Initialize a feature transformer.

        Parameters
        ----------
        input_dim : int
            Input size
        output_dim : int
            Output_size
        shared_layers : torch.nn.ModuleList
            The shared block that should be common to every step
        n_glu_independent : int
            Number of independent GLU layers
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization within GLU block(s)
        momentum : float
            Float value between 0 and 1 which will be used for momentum in batch norm
        """

        params = {
            "n_glu": n_glu_independent,
            "virtual_batch_size": virtual_batch_size,
            "momentum": momentum,
        }

        if shared_layers is None:
            # no shared layers
            self.shared = torch.nn.Identity()
            is_first = True
        else:
            self.shared = GLU_Block(
                input_dim,
                output_dim,
                first=True,
                shared_layers=shared_layers,
                n_glu=len(shared_layers),
                virtual_batch_size=virtual_batch_size,
                momentum=momentum,
            )
            is_first = False

        if n_glu_independent == 0:
            # no independent layers
            self.specifics = torch.nn.Identity()
        else:
            spec_input_dim = input_dim if is_first else output_dim
            self.specifics = GLU_Block(
                spec_input_dim,
                output_dim,
                first=is_first,
                **params,  # type: ignore
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shared(x)
        x = self.specifics(x)
        return x
