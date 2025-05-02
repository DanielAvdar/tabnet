# Empty file for PredictDataset class
from typing import Union

import torch
from torch.utils.data import Dataset

from ..error_handlers.data import model_input_data_check
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

        else:
            model_input_data_check(x)
            self.x = torch.from_numpy(x)
        self.x = self.x.float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> torch.Tensor:
        x = self.x[index]
        return x
