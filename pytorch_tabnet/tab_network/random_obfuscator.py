"""RandomObfuscator class module."""

import torch


class RandomObfuscator(torch.nn.Module):
    """Create and applies obfuscation masks.
    The obfuscation is done at group level to match attention.
    """

    def __init__(self, pretraining_ratio: float, group_matrix: torch.Tensor):
        """This create random obfuscation for self suppervised pretraining.

        Parameters.
        ----------
        pretraining_ratio : float
            Ratio of feature to randomly discard for reconstruction

        """
        super(RandomObfuscator, self).__init__()
        self.pretraining_ratio = pretraining_ratio
        # group matrix is set to boolean here to pass all posssible information
        self.group_matrix = (group_matrix > 0) + 0.0
        self.num_groups = group_matrix.shape[0]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate random obfuscation mask.

        Returns
        -------
        masked input and obfuscated variables.

        """
        bs = x.shape[0]

        obfuscated_groups = torch.bernoulli(self.pretraining_ratio * torch.ones((bs, self.num_groups), device=x.device))
        obfuscated_vars = torch.matmul(obfuscated_groups, self.group_matrix)
        masked_input = torch.mul(1 - obfuscated_vars, x)
        return masked_input, obfuscated_groups, obfuscated_vars
