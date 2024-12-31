import numpy as np
import pytest
from scipy import sparse as sparse
from torch.utils.data import DataLoader

import pytorch_tabnet
from pytorch_tabnet.pretraining_utils import create_dataloaders

from pytorch_tabnet.utils import validate_eval_set
from tests.test_pretraining_utils import mock_create_sampler


@pytest.mark.parametrize(
    "eval_set, eval_name, error_message",
    [
        (
                [(np.random.rand(20, 15), np.random.rand(20))],
                ["validation"],
                "Number of columns is different between X_validation .* and X_train .*",
        ),
        (
                [(np.random.rand(20, 10), np.random.rand(20, 2))],
                ["validation"],
                "Dimension mismatch between y_validation .* and y_train .*",
        ),
        (
                [(np.random.rand(25, 10), np.random.rand(20))],
                ["validation"],
                "You need the same number of rows between X_validation .* and y_validation .*",
        ),
        (
                [(np.random.rand(20, 10), np.random.rand(20))],
                ["validation", "extra_name"],
                "eval_set and eval_name have not the same length",
        ),
        (
                [(np.random.rand(20, 10))],
                ["validation"],
                "Each tuple of eval_set need to have two elements",
        ),
    ],
)
def test_validate_eval_set_errors(eval_set, eval_name, error_message):
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)

    with pytest.raises(AssertionError, match=error_message):
        validate_eval_set(eval_set, eval_name, X_train, y_train)


@pytest.mark.parametrize(
    "eval_name, expected_names",
    [
        (["validation"], ["validation"]),
        (None, ["val_0"])
    ]
)
def test_validate_eval_set_valid_inputs_and_default_eval_name(eval_name, expected_names):
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)
    eval_set = [(np.random.rand(20, 10), np.random.rand(20))]

    validated_names, validated_set = validate_eval_set(eval_set, eval_name, X_train, y_train)

    assert validated_names == expected_names
    assert len(validated_set) == len(eval_set)
    assert np.array_equal(validated_set[0][0], eval_set[0][0])
    assert np.array_equal(validated_set[0][1], eval_set[0][1])


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
