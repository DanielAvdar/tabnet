# Empty file for TorchDataset class
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..error_handlers.data import model_input_data_check, model_target_check


class TorchDataset(Dataset):
    """Format for numpy array.

    Parameters
    ----------
    X : 2D array
        The input matrix
    y : 2D array
        The one-hot encoded target

    """

    def __init__(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        model_input_data_check(x)
        self.x = torch.from_numpy(x).float()
        if y is not None:
            model_target_check(y)
            self.y = torch.from_numpy(y).float()
        else:
            self.y = None

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.y is None:
            return (self.x[index], None)
        else:
            x, y = self.x[index], self.y[index]
            return x, y
