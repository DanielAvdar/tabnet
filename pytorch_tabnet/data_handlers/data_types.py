from typing import Union

import numpy as np
import scipy
import torch

X_type = Union[np.ndarray, scipy.sparse.csr_matrix]
tn_type = Union[torch.Tensor, None]
