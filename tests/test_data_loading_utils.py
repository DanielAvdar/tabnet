import pytest
from unittest.mock import MagicMock
import scipy.sparse as sparse
import numpy as np
from torch.utils.data import DataLoader

from pytorch_tabnet.pretraining_utils import validate_eval_set, create_dataloaders
import pytorch_tabnet
# Mocking create_sampler for controlled testing
def mock_create_sampler(weights, X_train):
    need_shuffle = weights is not None
    sampler = MagicMock()
    return need_shuffle, sampler


# Mocking check_input to avoid its side effects
def mock_check_input(X):
    pass


# Test cases for create_dataloaders
@pytest.mark.parametrize(
    "X_train_sparse,eval_set_sparse,weights,batch_size,num_workers,drop_last,pin_memory",
    [

        (
            False,
            True,
            [0.2] * 100,
            128,
            1,
            False,
            False
        ),
    ],
)
def test_create_dataloaders(
    X_train_sparse,
    eval_set_sparse,
    weights,
    batch_size,
    num_workers,
    drop_last,
    pin_memory,
    monkeypatch,
):

    monkeypatch.setattr(pytorch_tabnet.utils, "create_sampler", mock_create_sampler)

    X_train = sparse.random(100, 10) if X_train_sparse else np.random.rand(100, 10)
    eval_set = (
        [sparse.random(50, 10), sparse.random(50, 10)]
        if eval_set_sparse
        else [np.random.rand(50, 10), np.random.rand(50, 10)]
    )

    train_dataloader, valid_dataloaders = create_dataloaders(
        X_train, eval_set, weights, batch_size, num_workers, drop_last, pin_memory
    )

    assert isinstance(train_dataloader, DataLoader)
    assert isinstance(valid_dataloaders, list)
    assert all(isinstance(loader, DataLoader) for loader in valid_dataloaders)

# Test cases for validate_eval_set


@pytest.mark.parametrize(
    "eval_set, eval_name, expected_eval_names",
    [
        (
            [np.random.rand(50, 10), np.random.rand(50, 10)],
            ["val_1", "val_2"],
            ["val_1", "val_2"],
        ),  # eval_name provided
        (
            [np.random.rand(50, 10)],
            None,
            ["val_0"],
        ),  # eval_name not provided
    ],
)
def test_validate_eval_set(eval_set, eval_name, expected_eval_names, monkeypatch):
    monkeypatch.setattr(pytorch_tabnet.utils, "check_input", mock_check_input)

    X_train = np.random.rand(100, 10)
    eval_names = validate_eval_set(eval_set, eval_name, X_train)

    assert eval_names == expected_eval_names


def test_validate_eval_set_mismatched_lengths():
    eval_set = [np.random.rand(50, 10)]
    eval_name = ["val_1", "val_2"]
    X_train = np.random.rand(100, 10)
    with pytest.raises(AssertionError):
        validate_eval_set(eval_set, eval_name, X_train)


def test_validate_eval_set_mismatched_columns(monkeypatch):
    monkeypatch.setattr(pytorch_tabnet.utils, "check_input", mock_check_input)

    eval_set = [np.random.rand(50, 5)]
    X_train = np.random.rand(100, 10)
    with pytest.raises(AssertionError):
        validate_eval_set(eval_set, None, X_train)