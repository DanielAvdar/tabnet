# Empty file for TorchDataset class
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from pytorch_tabnet.error_handlers.data import model_input_data_check, model_target_check


class TorchDataset(Dataset):
    """Format for numpy array.

    Parameters
    ----------
    X : 2D array
        The input matrix
    y : 2D array
        The one-hot encoded target

    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        model_input_data_check(x)
        model_target_check(y)
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.x[index], self.y[index]
        return x, y
