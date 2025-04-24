"""TabNetDecoder class module."""

from typing import List

import torch
from torch.nn import Linear

from .feat_transformer import FeatTransformer
from .utils_funcs import initialize_non_glu


class TabNetDecoder(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_d: int = 8,
        n_steps: int = 3,
        n_independent: int = 1,
        n_shared: int = 1,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
    ):
        """Defines main part of the TabNet network without the embedding layers.

        Parameters
        ----------
        input_dim : int
            Number of features
        output_dim : int or list of int for multi task classification
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        n_steps : int
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        n_independent : int
            Number of independent GLU layer in each GLU block (default 1)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 1)
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm

        """
        super(TabNetDecoder, self).__init__()
        self.input_dim = input_dim
        self.n_d = n_d
        self.n_steps = n_steps
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size

        self.feat_transformers = torch.nn.ModuleList()

        if self.n_shared > 0:
            shared_feat_transform = torch.nn.ModuleList()
            for _i in range(self.n_shared):
                shared_feat_transform.append(Linear(n_d, 2 * n_d, bias=False))
        else:
            shared_feat_transform = None

        for _step in range(n_steps):
            transformer = FeatTransformer(
                n_d,
                n_d,
                shared_feat_transform,
                n_glu_independent=self.n_independent,
                virtual_batch_size=self.virtual_batch_size,
                momentum=momentum,
            )
            self.feat_transformers.append(transformer)

        self.reconstruction_layer = Linear(n_d, self.input_dim, bias=False)
        initialize_non_glu(self.reconstruction_layer, n_d, self.input_dim)

    def forward(self, steps_output: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass for TabNetDecoder.

        Parameters
        ----------
        steps_output : List[torch.Tensor]
            List of tensors from each step of the encoder.

        Returns
        -------
        torch.Tensor
            Reconstructed input tensor.

        """
        res = 0
        for step_nb, step_output in enumerate(steps_output):
            x = self.feat_transformers[step_nb](step_output)
            res = torch.add(res, x)
        res = self.reconstruction_layer(res)
        return res
