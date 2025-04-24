"""GLU_Block class module."""

from typing import Optional

import torch

# from pytorch_tabnet.tab_network.glu_layer import GLU_Layer
from .glu_layer import GLU_Layer


class GLU_Block(torch.nn.Module):
    """Independent GLU block, specific to each step."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_glu: int = 2,
        first: bool = False,
        shared_layers: Optional[torch.nn.ModuleList] = None,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
    ):
        super(GLU_Block, self).__init__()
        self.first = first
        self.shared_layers = shared_layers
        self.n_glu = n_glu
        self.glu_layers = torch.nn.ModuleList()

        params = {"virtual_batch_size": virtual_batch_size, "momentum": momentum}

        fc = shared_layers[0] if shared_layers else None
        self.glu_layers.append(GLU_Layer(input_dim, output_dim, fc=fc, **params))  # type: ignore
        for glu_id in range(1, self.n_glu):
            fc = shared_layers[glu_id] if shared_layers else None
            self.glu_layers.append(GLU_Layer(output_dim, output_dim, fc=fc, **params))  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.sqrt(torch.tensor([0.5], device=x.device).float())
        if self.first:  # the first layer of the block has no scale multiplication
            x = self.glu_layers[0](x)
            layers_left = range(1, self.n_glu)
        else:
            layers_left = range(self.n_glu)

        for glu_id in layers_left:
            x = torch.add(x, self.glu_layers[glu_id](x))
            x = x * scale
        return x
