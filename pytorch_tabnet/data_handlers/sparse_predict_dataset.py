# Empty file for SparsePredictDataset class
import scipy
import torch
from torch.utils.data import Dataset


class SparsePredictDataset(Dataset):
    """Format for csr_matrix.

    Parameters
    ----------
    X : CSR matrix
        The input matrix

    """

    def __init__(self, x: scipy.sparse.csr_matrix):
        self.x = torch.from_numpy(x.toarray()).float()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        x = self.x[index]
        return x
