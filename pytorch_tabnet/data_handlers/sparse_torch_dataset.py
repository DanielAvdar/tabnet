# Empty file for SparseTorchDataset class
from typing import Tuple

import numpy as np
import scipy
import torch
from torch.utils.data import Dataset


class SparseTorchDataset(Dataset):
    """Format for csr_matrix.

    Parameters
    ----------
    X : CSR matrix
        The input matrix
    y : 2D array
        The one-hot encoded target

    """

    def __init__(self, x: scipy.sparse.csr_matrix, y: np.ndarray):
        self.x = torch.from_numpy(x.toarray()).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.x[index]
        y = self.y[index]
        return x, y
