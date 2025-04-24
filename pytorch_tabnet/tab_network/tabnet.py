"""TabNet main class module."""

from typing import List, Optional, Union

import torch

from .embedding_generator import EmbeddingGenerator

# from pytorch_tabnet.tab_network import TabNetNoEmbeddings
# from pytorch_tabnet.tab_network.embedding_generator import EmbeddingGenerator
from .tabnet_noembeddings import TabNetNoEmbeddings


class TabNet(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: Union[int, List[int]],
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        gamma: float = 1.3,
        cat_idxs: List[int] = None,
        cat_dims: List[int] = None,
        cat_emb_dim: Union[int, List[int]] = 1,
        n_independent: int = 2,
        n_shared: int = 2,
        epsilon: float = 1e-15,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
        mask_type: str = "sparsemax",
        group_attention_matrix: Optional[torch.Tensor] = None,
    ):
        """Defines TabNet network.

        Parameters
        ----------
        input_dim : int
            Initial number of features
        output_dim : int
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        n_a : int
            Dimension of the attention  layer (usually between 4 and 64)
        n_steps : int
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        cat_idxs : list of int
            Index of each categorical column in the dataset
        cat_dims : list of int
            Number of categories in each categorical column
        cat_emb_dim : int or list of int
            Size of the embedding of categorical features
            if int, all categorical features will have same embedding size
            if list of int, every corresponding feature will have specific size
        n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        epsilon : float
            Avoid log(0), this should be kept very low
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        group_attention_matrix : torch matrix
            Matrix of size (n_groups, input_dim), m_ij = importance within group i of feature j

        """
        if cat_dims is None:
            cat_dims = []
        if cat_idxs is None:
            cat_idxs = []
        super(TabNet, self).__init__()
        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        self.cat_emb_dim = cat_emb_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type

        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independent can't be both zero.")

        self.virtual_batch_size = virtual_batch_size
        self.embedder = EmbeddingGenerator(input_dim, cat_dims, cat_idxs, cat_emb_dim, group_attention_matrix)
        self.post_embed_dim = self.embedder.post_embed_dim

        self.tabnet = TabNetNoEmbeddings(
            self.post_embed_dim,
            output_dim,
            n_d,
            n_a,
            n_steps,
            gamma,
            n_independent,
            n_shared,
            epsilon,
            virtual_batch_size,
            momentum,
            mask_type,
            self.embedder.embedding_group_matrix,
        )

    def forward(self, x: torch.Tensor) -> tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]:
        """Forward pass for TabNet.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]
            Output tensor(s) and mask loss.

        """
        x = self.embedder(x)
        return self.tabnet(x)

    def forward_masks(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Return feature masks for each step in TabNet.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        tuple[torch.Tensor, dict]
            Tuple of (explanation mask, step-wise masks dictionary).

        """
        x = self.embedder(x)
        return self.tabnet.forward_masks(x)
