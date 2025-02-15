import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from pytorch_tabnet.utils import check_input

X_type = Union[np.ndarray, scipy.sparse.csr_matrix]
tn_type = Union[torch.Tensor, None]


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
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
        self.x = torch.from_numpy(x.toarray()).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.x[index]
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


class SparsePredictDataset(Dataset):
    """
    Format for csr_matrix

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


@dataclass
class TBDataLoader:
    name: str
    dataset: Union[PredictDataset, TorchDataset]
    batch_size: int
    weights: Optional[torch.Tensor] = None
    pre_training: bool = False
    drop_last: bool = False
    pin_memory: bool = False
    predict: bool = False
    all_at_once: bool = False

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, tn_type, tn_type]]:
        if self.all_at_once:
            if self.pre_training or isinstance(self.dataset, PredictDataset) or isinstance(self.dataset, SparsePredictDataset):
                yield self.dataset.x, None, None
            else:
                yield self.dataset.x, self.dataset.y, None
            return
        ds_len = len(self.dataset)
        perm = None
        if not self.predict:
            perm = torch.randperm(ds_len, pin_memory=self.pin_memory)
        batched_starts = [i for i in range(0, ds_len, self.batch_size)]
        batched_starts += [0] if len(batched_starts) == 0 else []
        for start in batched_starts[: len(self)]:
            if self.predict:
                yield self.make_predict_batch(ds_len, start)
            else:
                yield self.make_train_batch(ds_len, perm, start)

    def make_predict_batch(self, ds_len: int, start: int) -> Tuple[torch.Tensor, tn_type, None]:
        end_at = start + self.batch_size
        if end_at > ds_len:
            end_at = ds_len
        x, y, w = None, None, None
        if self.pre_training or isinstance(self.dataset, PredictDataset) or isinstance(self.dataset, SparsePredictDataset):
            # return self.dataset.x[start:end_at], None, None
            x = self.dataset.x[start:end_at]

        else:
            x, y = self.dataset.x[start:end_at], self.dataset.y[start:end_at]
        w = None if self.weights is None else self.weights[start:end_at]

        return x, y, w

    def make_train_batch(self, ds_len: int, perm: Optional[torch.Tensor], start: int) -> Tuple[torch.Tensor, tn_type, tn_type]:
        end_at = start + self.batch_size
        left_over = None
        if end_at > ds_len:
            left_over = end_at - ds_len
            end_at = ds_len
        indexes = perm[start:end_at]
        x, y, w = None, None, None
        if self.pre_training:
            x = self.dataset.x[indexes]
        else:
            x, y = self.dataset.x[indexes], self.dataset.y[indexes]
        w = self.get_weights(indexes)

        if left_over is not None:
            lo_indexes = perm[:left_over]

            if self.pre_training:
                x = torch.cat((x, self.dataset.x[lo_indexes]))
                w = self.get_weights(lo_indexes)
            else:
                x = torch.cat((x, self.dataset.x[lo_indexes]))
                y = torch.cat((y, self.dataset.y[lo_indexes]))
                w = None if self.weights is None else torch.cat((w, self.get_weights(lo_indexes)))
        return x, y, w

    def __len__(self) -> int:
        res = math.ceil(len(self.dataset) / self.batch_size)
        need_to_drop_last = self.drop_last and not self.predict
        need_to_drop_last = need_to_drop_last and (res > 1)
        res -= need_to_drop_last
        return res

    def get_weights(self, i: torch.Tensor = None) -> Union[torch.Tensor, None]:
        if self.weights is None:
            return None
        if i is None:
            return self.weights
        return self.weights[i]


def create_dataloaders(
    X_train: X_type,
    y_train: np.ndarray,
    eval_set: List[Tuple[X_type, np.ndarray]],
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
    _need_shuffle, _sampler = create_sampler(weights, y_train)
    t_weights = None
    # if isinstance(weights, int) and weights == 1:
    #     t_weights = create_class_weights(y_train,)
    if isinstance(weights, np.ndarray):
        t_weights = torch.from_numpy(weights)

    if scipy.sparse.issparse(X_train):
        train_dataloader = TBDataLoader(
            name="train-data",
            dataset=SparseTorchDataset(X_train.astype(np.float32), y_train),
            batch_size=batch_size,
            weights=t_weights,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )
    else:
        train_dataloader = TBDataLoader(
            name="train-data",
            dataset=TorchDataset(X_train.astype(np.float32), y_train),
            batch_size=batch_size,
            weights=t_weights,
            # sampler=sampler,
            # shuffle=need_shuffle,
            # num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )

    valid_dataloaders = []
    for X, y in eval_set:
        v_t_weights = None

        if scipy.sparse.issparse(X):
            valid_dataloaders.append(
                TBDataLoader(
                    name="val-data",
                    dataset=SparseTorchDataset(X.astype(np.float32), y),
                    batch_size=batch_size,
                    weights=v_t_weights,
                    pin_memory=pin_memory,
                    predict=True,
                    # all_at_once=True,
                )
            )
        else:
            valid_dataloaders.append(
                TBDataLoader(
                    name="val-data",
                    dataset=TorchDataset(X.astype(np.float32), y),
                    batch_size=batch_size,
                    weights=v_t_weights,
                    pin_memory=pin_memory,
                    predict=True,
                    # all_at_once=True,
                )
            )

    return train_dataloader, valid_dataloaders


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
            samples_weight = create_class_weights(y_train)
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


def create_class_weights(y_train: torch.Tensor, base_size: float = 1.0) -> torch.Tensor:
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weights_ = base_size / class_sample_count
    samples_weight = np.zeros(len(y_train))
    for i, t in enumerate(np.unique(y_train)):
        samples_weight[y_train == t] = weights_[i]
    # samples_weight = np.array([weights_[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    return samples_weight


def create_dataloaders_pt(
    X_train: Union[scipy.sparse.csr_matrix, np.ndarray],
    eval_set: List[Union[scipy.sparse.csr_matrix, np.ndarray]],
    weights: Union[int, Dict[Any, Any], Iterable[Any]],
    batch_size: int,
    num_workers: int,
    drop_last: bool,
    pin_memory: bool,
) -> Tuple[DataLoader, List[DataLoader]]:
    """
    Create dataloaders with or without subsampling depending on weights and balanced.

    Parameters
    ----------
    X_train : np.ndarray or scipy.sparse.csr_matrix
        Training data
    eval_set : list of np.array (for Xs and ys) or scipy.sparse.csr_matrix (for Xs)
        List of eval sets
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
    _need_shuffle, _sampler = create_sampler(weights, X_train)

    if scipy.sparse.issparse(X_train):
        train_dataloader = TBDataLoader(
            name="train-data",
            dataset=SparsePredictDataset(X_train),
            batch_size=batch_size,
            # sampler=sampler,
            # shuffle=need_shuffle,
            # num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
            pre_training=True,
        )
    else:
        train_dataloader = TBDataLoader(
            name="train-data",
            dataset=PredictDataset(X_train),
            batch_size=batch_size,
            # sampler=sampler,
            # shuffle=need_shuffle,
            # num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
            pre_training=True,
        )

    valid_dataloaders = []
    for X in eval_set:
        if scipy.sparse.issparse(X):
            valid_dataloaders.append(
                TBDataLoader(
                    name="val-data",
                    dataset=SparsePredictDataset(X),
                    batch_size=batch_size,
                    # sampler=sampler,
                    # shuffle=need_shuffle,
                    # num_workers=num_workers,
                    drop_last=drop_last,
                    pin_memory=pin_memory,
                    predict=True,
                    # all_at_once=True,
                )
            )
        else:
            valid_dataloaders.append(
                TBDataLoader(
                    name="val-data",
                    dataset=PredictDataset(X),
                    batch_size=batch_size,
                    # sampler=sampler,
                    # shuffle=need_shuffle,
                    # num_workers=num_workers,
                    drop_last=drop_last,
                    pin_memory=pin_memory,
                    predict=True,
                    # all_at_once=True,
                )
            )

    return train_dataloader, valid_dataloaders


def validate_eval_set(eval_set: List[np.ndarray], eval_name: List[str], X_train: np.ndarray) -> List[str]:
    """Check if the shapes of eval_set are compatible with X_train.

    Parameters
    ----------
    eval_set : List of numpy array
        The list evaluation set.
        The last one is used for early stopping
    eval_name : List[str]
        Names for eval sets.
    X_train : np.ndarray
        Train owned products

    Returns
    -------
    eval_names : list of str
        Validated list of eval_names.

    """
    eval_names = eval_name or [f"val_{i}" for i in range(len(eval_set))]
    assert len(eval_set) == len(eval_names), "eval_set and eval_name have not the same length"

    for set_nb, X in enumerate(eval_set):
        check_input(X)
        msg = f"Number of columns is different between eval set {set_nb}" + f"({X.shape[1]}) and X_train ({X_train.shape[1]})"
        assert X.shape[1] == X_train.shape[1], msg
    return eval_names
