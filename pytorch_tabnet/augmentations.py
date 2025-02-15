from typing import Tuple

import numpy as np
import torch

from pytorch_tabnet.utils import define_device


class RegressionSMOTE:
    """
    Apply SMOTE

    This will average a percentage p of the elements in the batch with other elements.
    The target will be averaged as well (this might work with binary classification
    and certain loss), following a beta distribution.
    """

    def __init__(
        self,
        device_name: str = "auto",
        p: float = 0.8,
        alpha: float = 0.5,
        beta: float = 0.5,
        seed: int = 0,
    ):
        ""
        self.seed = seed
        self._set_seed()
        self.device = define_device(device_name)
        self.alpha = alpha
        self.beta = beta
        self.p = p
        if (p < 0.0) or (p > 1.0):
            raise ValueError("Value of p should be between 0. and 1.")

    def _set_seed(self) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        return

    def __call__(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = X.shape[0]
        random_values = torch.rand(batch_size, device=self.device)
        idx_to_change = random_values < self.p

        # ensure that first element to switch has probability > 0.5
        np_betas = np.random.beta(self.alpha, self.beta, batch_size) / 2 + 0.5
        random_betas = torch.from_numpy(np_betas).to(self.device).float()
        index_permute = torch.randperm(batch_size, device=self.device)

        X[idx_to_change] = random_betas[idx_to_change, None] * X[idx_to_change]
        X[idx_to_change] += (1 - random_betas[idx_to_change, None]) * X[index_permute][idx_to_change].view(X[idx_to_change].size())  # noqa

        y[idx_to_change] = random_betas[idx_to_change, None] * y[idx_to_change]
        y[idx_to_change] += (1 - random_betas[idx_to_change, None]) * y[index_permute][idx_to_change].view(y[idx_to_change].size())  # noqa

        return X, y


class ClassificationSMOTE:
    """
    Apply SMOTE for classification tasks.

    This will average a percentage p of the elements in the batch with other elements.
    The target will stay unchanged and keep the value of the most important row in the mix.
    """

    def __init__(
        self,
        device_name: str = "auto",
        p: float = 0.8,
        alpha: float = 0.5,
        beta: float = 0.5,
        seed: int = 0,
    ):
        ""
        self.seed = seed
        self._set_seed()
        self.device = define_device(device_name)
        self.alpha = alpha
        self.beta = beta
        self.p = p
        if (p < 0.0) or (p > 1.0):
            raise ValueError("Value of p should be between 0. and 1.")

    def _set_seed(self) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        return

    def __call__(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = X.shape[0]
        random_values = torch.rand(batch_size, device=self.device)
        idx_to_change = random_values < self.p

        # ensure that first element to switch has probability > 0.5
        np_betas = np.random.beta(self.alpha, self.beta, batch_size) / 2 + 0.5
        random_betas = torch.from_numpy(np_betas).to(self.device).float()
        index_permute = torch.randperm(batch_size, device=self.device)

        X[idx_to_change] = random_betas[idx_to_change, None] * X[idx_to_change]
        X[idx_to_change] += (1 - random_betas[idx_to_change, None]) * X[index_permute][idx_to_change].view(X[idx_to_change].size())  # noqa

        return X, y
