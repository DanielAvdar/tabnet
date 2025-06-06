# Unified dataset implementation for both prediction and training use cases
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from ..error_handlers.data import model_input_data_check, model_target_check
from .data_types import X_type


class UnifiedDataset(Dataset):
    """Unified dataset class that supports both prediction and training use cases.

    When y is None, behaves like PredictDataset (returns only x).
    When y is provided, behaves like TorchDataset (returns x, y).

    Parameters
    ----------
    x : Union[X_type, torch.Tensor]
        The input matrix (2D array or torch tensor)
    y : Optional[np.ndarray]
        The target matrix (2D array), None for prediction use case

    """

    def __init__(self, x: Union[X_type, torch.Tensor], y: Optional[np.ndarray] = None):
        # Handle x input (similar to original PredictDataset)
        if isinstance(x, torch.Tensor):
            self.x = x
        else:
            model_input_data_check(x)
            self.x = torch.from_numpy(x)
        self.x = self.x.float()

        # Handle y input (similar to original TorchDataset when y is provided)
        if y is not None:
            model_target_check(y)
            self.y = torch.from_numpy(y).float()
        else:
            self.y = None

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.x[index]
        if self.y is None:
            # PredictDataset behavior: return only x
            return x
        else:
            # TorchDataset behavior: return x, y
            y = self.y[index]
            return x, y
