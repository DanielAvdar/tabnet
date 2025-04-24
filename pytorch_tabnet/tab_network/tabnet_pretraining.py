"""TabNetPretraining class module."""

from typing import List, Optional

import torch

from .embedding_generator import EmbeddingGenerator
from .random_obfuscator import RandomObfuscator
from .tabnet_decoder import TabNetDecoder

# from pytorch_tabnet.tab_network import TabNetEncoder, TabNetDecoder
# from pytorch_tabnet.tab_network.random_obfuscator import RandomObfuscator
# from pytorch_tabnet.tab_network.embedding_generator import EmbeddingGenerator
from .tabnet_encoder import TabNetEncoder


class TabNetPretraining(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        pretraining_ratio: float = 0.2,
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        gamma: float = 1.3,
        cat_idxs: List[int] = None,
        cat_dims: List[int] = None,
        cat_emb_dim: int = 1,
        n_independent: int = 2,
        n_shared: int = 2,
        epsilon: float = 1e-15,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
        mask_type: str = "sparsemax",
        n_shared_decoder: int = 1,
        n_indep_decoder: int = 1,
        group_attention_matrix: Optional[torch.Tensor] = None,
    ):
        if cat_dims is None:
            cat_dims = []
        if cat_idxs is None:
            cat_idxs = []
        super(TabNetPretraining, self).__init__()

        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        self.cat_emb_dim = cat_emb_dim

        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type
        self.pretraining_ratio = pretraining_ratio
        self.n_shared_decoder = n_shared_decoder
        self.n_indep_decoder = n_indep_decoder

        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independent can't be both zero.")

        self.virtual_batch_size = virtual_batch_size
        self.embedder = EmbeddingGenerator(input_dim, cat_dims, cat_idxs, cat_emb_dim, group_attention_matrix)
        self.post_embed_dim = self.embedder.post_embed_dim

        self.masker = RandomObfuscator(self.pretraining_ratio, group_matrix=self.embedder.embedding_group_matrix)
        self.encoder = TabNetEncoder(
            input_dim=self.post_embed_dim,
            output_dim=self.post_embed_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            mask_type=mask_type,
            group_attention_matrix=self.embedder.embedding_group_matrix,
        )
        self.decoder = TabNetDecoder(
            self.post_embed_dim,
            n_d=n_d,
            n_steps=n_steps,
            n_independent=self.n_indep_decoder,
            n_shared=self.n_shared_decoder,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: res, embedded_x, obf_vars
        res : output of reconstruction
        embedded_x : embedded input
        obf_vars : which variable where obfuscated.
        """
        embedded_x = self.embedder(x)
        if self.training:
            masked_x, obfuscated_groups, obfuscated_vars = self.masker(embedded_x)
            # set prior of encoder with obfuscated groups
            prior = 1 - obfuscated_groups
            steps_out, _ = self.encoder(masked_x, prior=prior)
            res = self.decoder(steps_out)
            return res, embedded_x, obfuscated_vars
        else:
            steps_out, _ = self.encoder(embedded_x)
            res = self.decoder(steps_out)
            return (res, embedded_x, torch.ones(embedded_x.shape, device=x.device))

    def forward_masks(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        embedded_x = self.embedder(x)
        return self.encoder.forward_masks(embedded_x)
