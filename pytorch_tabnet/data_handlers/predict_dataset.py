# Empty file for PredictDataset class
from typing import Union

import scipy
import torch
from torch.utils.data import Dataset

from .data_types import X_type


class PredictDataset(Dataset):
    """Format for numpy array.

    Parameters
    ----------
    X : 2D array
        The input matrix

    """

    def __init__(self, x: Union[X_type, torch.Tensor]):
        if isinstance(x, torch.Tensor):
            self.x = x
        elif scipy.sparse.issparse(x):
            self.x = torch.from_numpy(x.toarray())
        else:
            self.x = torch.from_numpy(x)
        self.x = self.x.float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> torch.Tensor:
        x = self.x[index]
        return x
