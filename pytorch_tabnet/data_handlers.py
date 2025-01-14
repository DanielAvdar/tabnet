from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class TorchDataset(Dataset):
    """
    Format for numpy array

    Parameters
    ----------
    X : 2D array
        The input matrix
    y : 2D array
        The one-hot encoded target
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        x, y = self.x[index], self.y[index]
        return x, y


class SparseTorchDataset(Dataset):
    """
    Format for csr_matrix

    Parameters
    ----------
    X : CSR matrix
        The input matrix
    y : 2D array
        The one-hot encoded target
    """

    def __init__(self, x: scipy.sparse.csr_matrix, y: np.ndarray):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, np.ndarray]:
        x = torch.from_numpy(self.x[index].toarray()[0]).float()
        y = self.y[index]
        return x, y


class PredictDataset(Dataset):
    """
    Format for numpy array

    Parameters
    ----------
    X : 2D array
        The input matrix
    """

    def __init__(self, x: np.ndarray):
        self.x = x

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> np.ndarray:
        x = self.x[index]
        return x


class SparsePredictDataset(Dataset):
    """
    Format for csr_matrix

    Parameters
    ----------
    X : CSR matrix
        The input matrix
    """

    def __init__(self, x: scipy.sparse.csr_matrix):
        self.x = x

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        x = torch.from_numpy(self.x[index].toarray()[0]).float()
        return x


def create_sampler(weights: Union[int, Dict, Iterable], y_train: np.ndarray) -> Tuple[bool, Optional[WeightedRandomSampler]]:
    """
    This creates a sampler from the given weights

    Parameters
    ----------
    weights : either 0, 1, dict or iterable
        if 0 (default) : no weights will be applied
        if 1 : classification only, will balanced class with inverse frequency
        if dict : keys are corresponding class values are sample weights
        if iterable : list or np array must be of length equal to nb elements
                      in the training set
    y_train : np.array
        Training targets
    """
    if isinstance(weights, int):
        if weights == 0:
            need_shuffle = True
            sampler = None
        elif weights == 1:
            need_shuffle = False
            class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])

            weights_ = 1.0 / class_sample_count

            samples_weight = np.array([weights_[t] for t in y_train])

            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        else:
            raise ValueError("Weights should be either 0, 1, dictionnary or list.")
    elif isinstance(weights, dict):
        # custom weights per class
        need_shuffle = False
        samples_weight = np.array([weights[t] for t in y_train])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    else:
        # custom weights
        if len(weights) != len(y_train):  # type: ignore
            raise ValueError("Custom weights should match number of train samples.")
        need_shuffle = False
        samples_weight = np.array(weights)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return need_shuffle, sampler


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    eval_set: List[Tuple[np.ndarray, np.ndarray]],
    weights: Union[int, Dict, Iterable],
    batch_size: int,
    num_workers: int,
    drop_last: bool,
    pin_memory: bool,
) -> Tuple[DataLoader, List[DataLoader]]:
    """
    Create dataloaders with or without subsampling depending on weights and balanced.

    Parameters
    ----------
    X_train : np.ndarray
        Training data
    y_train : np.array
        Mapped Training targets
    eval_set : list of tuple
        List of eval tuple set (X, y)
    weights : either 0, 1, dict or iterable
        if 0 (default) : no weights will be applied
        if 1 : classification only, will balanced class with inverse frequency
        if dict : keys are corresponding class values are sample weights
        if iterable : list or np array must be of length equal to nb elements
                      in the training set
    batch_size : int
        how many samples per batch to load
    num_workers : int
        how many subprocesses to use for data loading. 0 means that the data
        will be loaded in the main process
    drop_last : bool
        set to True to drop the last incomplete batch, if the dataset size is not
        divisible by the batch size. If False and the size of dataset is not
        divisible by the batch size, then the last batch will be smaller
    pin_memory : bool
        Whether to pin GPU memory during training

    Returns
    -------
    train_dataloader, valid_dataloader : torch.DataLoader, torch.DataLoader
        Training and validation dataloaders
    """
    need_shuffle, sampler = create_sampler(weights, y_train)

    if scipy.sparse.issparse(X_train):
        train_dataloader = DataLoader(
            SparseTorchDataset(X_train.astype(np.float32), y_train),
            batch_size=batch_size,
            sampler=sampler,
            shuffle=need_shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )
    else:
        train_dataloader = DataLoader(
            TorchDataset(X_train.astype(np.float32), y_train),
            batch_size=batch_size,
            sampler=sampler,
            shuffle=need_shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )

    valid_dataloaders = []
    for X, y in eval_set:
        if scipy.sparse.issparse(X):
            valid_dataloaders.append(
                DataLoader(
                    SparseTorchDataset(X.astype(np.float32), y),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
            )
        else:
            valid_dataloaders.append(
                DataLoader(
                    TorchDataset(X.astype(np.float32), y),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
            )

    return train_dataloader, valid_dataloaders
