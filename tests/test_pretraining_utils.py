import pytest
from unittest.mock import MagicMock
import numpy as np

from pytorch_tabnet.pretraining_utils import validate_eval_set
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







####
import pytest

from pytorch_tabnet.utils import check_embedding_parameters


@pytest.mark.parametrize(
    "cat_dims, cat_idxs, cat_emb_dim, expected_output",
    [
        ([10, 20], [0, 1], 5, ([10, 20], [0, 1], [5, 5])),
        ([30, 40], [1, 0], [4, 6], ([40, 30], [1, 0], [6, 4])),
    ]
)
def test_check_embedding_parameters_valid(cat_dims, cat_idxs, cat_emb_dim, expected_output):
    assert check_embedding_parameters(cat_dims, cat_idxs, cat_emb_dim) == expected_output


@pytest.mark.parametrize(
    "cat_dims, cat_idxs, cat_emb_dim, error_message",
    [
        ([], [0, 1], 5, "If cat_idxs is non-empty, cat_dims must be defined as a list of same length."),
        ([10, 20], [], 5, "If cat_dims is non-empty, cat_idxs must be defined as a list of same length."),
        ([10], [0, 1], 5, "The lists cat_dims and cat_idxs must have the same length."),
        # ([10, 20], [0, 1], [5], 'cat_emb_dim and cat_dims must be lists of same length, got 1 and 2'),
    ]
)
def test_check_embedding_parameters_invalid(cat_dims, cat_idxs, cat_emb_dim, error_message):
    with pytest.raises(ValueError, match=error_message):
        check_embedding_parameters(cat_dims, cat_idxs, cat_emb_dim)




