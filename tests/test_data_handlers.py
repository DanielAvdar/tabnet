import math
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from scipy import sparse as sparse
from scipy.sparse import csr_matrix
from torch.utils.data import WeightedRandomSampler

import pytorch_tabnet
from pytorch_tabnet.data_handlers import TBDataLoader, create_dataloaders, create_dataloaders_pt, create_sampler, validate_eval_set
from pytorch_tabnet.utils import check_embedding_parameters


@pytest.fixture
def sample_data():
    X_train = np.random.rand(100, 20).astype(np.float32)
    y_train = np.random.randint(0, 2, size=100).astype(np.float32)
    eval_set = [
        (np.random.rand(20, 20).astype(np.float32), np.random.randint(0, 2, size=20).astype(np.float32)),
    ]
    return X_train, y_train, eval_set


@pytest.fixture
def sparse_sample_data():
    X_train = csr_matrix(np.random.rand(100, 20).astype(np.float32))
    y_train = np.random.randint(0, 2, size=100).astype(np.float32)
    eval_set = [
        (csr_matrix(np.random.rand(20, 20).astype(np.float32)), np.random.randint(0, 2, size=20).astype(np.float32)),
    ]
    return X_train, y_train, eval_set


def test_create_dataloaders_dense_data(sample_data):
    X_train, y_train, eval_set = sample_data
    batch_size = 16
    num_workers = 0
    drop_last = True
    pin_memory = False
    weights = 0

    train_loader, valid_loaders = create_dataloaders(X_train, y_train, eval_set, weights, batch_size, num_workers, drop_last, pin_memory)

    # assert isinstance(train_loader, DataLoader)
    assert len(train_loader) == len(y_train) // batch_size
    assert len(valid_loaders) == len(eval_set)
    for val_loader in valid_loaders:
        assert isinstance(val_loader, TBDataLoader)
        assert len(val_loader) > 0


def test_create_dataloaders_sparse_data(sparse_sample_data):
    X_train, y_train, eval_set = sparse_sample_data
    batch_size = 16
    num_workers = 0
    drop_last = False
    pin_memory = False
    weights = 0

    train_loader, valid_loaders = create_dataloaders(X_train, y_train, eval_set, weights, batch_size, num_workers, drop_last, pin_memory)

    # assert isinstance(train_loader, DataLoader)
    assert len(train_loader) == (len(y_train) + batch_size - 1) // batch_size
    assert len(valid_loaders) == len(eval_set)
    for val_loader in valid_loaders:
        assert isinstance(val_loader, TBDataLoader)


@pytest.mark.parametrize(
    "weights, X_train, y_train, eval_set",
    [
        (
            0,
            np.random.rand(100, 20).astype(np.float32),
            np.random.randint(0, 2, size=100).astype(np.float32),
            [(np.random.rand(20, 20).astype(np.float32), np.random.randint(0, 2, size=20).astype(np.float32))],
        ),
        (
            np.random.rand(100),
            np.random.rand(100, 20).astype(np.float32),
            np.random.randint(0, 2, size=100).astype(np.float32),
            [(np.random.rand(20, 20).astype(np.float32), np.random.randint(0, 2, size=20).astype(np.float32))],
        ),
        (
            np.random.rand(100),
            csr_matrix(np.random.rand(100, 20).astype(np.float32)),
            np.random.randint(0, 2, size=100).astype(np.float32),
            [(csr_matrix(np.random.rand(20, 20).astype(np.float32)), np.random.randint(0, 2, size=20).astype(np.float32))],
        ),
    ],
)
def test_create_dataloaders_weights_handling_dense(weights, X_train, y_train, eval_set):
    # X_train, y_train, eval_set = sample_data
    batch_size = 16
    num_workers = 0
    drop_last = False
    pin_memory = False

    train_loader, _ = create_dataloaders(
        X_train,
        y_train,
        eval_set,
        weights,
        batch_size,
        num_workers,
        drop_last,
        pin_memory,
    )

    # Check if shuffle matches the expected behavior based on weights
    data_loaded = [d for d in train_loader]
    assert len(data_loaded) - len(y_train) // batch_size >= 0
    assert len(data_loaded) - len(y_train) // batch_size <= 1
    assert len(data_loaded) == len(train_loader)
    for i, data in enumerate(data_loaded):
        assert data[1].shape[0] == batch_size, f"Batch size should be consistent at {i}th batch"


@pytest.mark.parametrize(
    "batch_size, drop_last, expected_batches",
    [
        (16, True, 6),  # Rounding down with drop_last=True
        (16, False, 7),  # Including the last partial batch with drop_last=False
    ],
)
def test_create_dataloaders_batch_behavior_dense(sample_data, batch_size, drop_last, expected_batches):
    X_train, y_train, eval_set = sample_data
    num_workers = 0
    pin_memory = False
    weights = 0

    train_loader, _ = create_dataloaders(X_train, y_train, eval_set, weights, batch_size, num_workers, drop_last, pin_memory)

    # Check if the correct number of batches is created
    assert len(train_loader) == expected_batches


@pytest.mark.parametrize(
    "batch_size, pin_memory, expected_pin_memory",
    [
        (32, True, True),
        (32, False, False),
    ],
)
def test_create_dataloaders_pin_memory_dense(sample_data, batch_size, pin_memory, expected_pin_memory):
    X_train, y_train, eval_set = sample_data
    num_workers = 0
    drop_last = False
    weights = 0

    train_loader, _ = create_dataloaders(X_train, y_train, eval_set, weights, batch_size, num_workers, drop_last, pin_memory)

    # Retrieve the actual pin_memory value from train_loader
    assert train_loader.pin_memory == expected_pin_memory


@pytest.mark.parametrize(
    "weights, y_train, expected_need_shuffle, expected_sampler_type",
    [
        (0, [0, 1, 0, 1], True, type(None)),
        (1, [0, 1, 0, 1], False, WeightedRandomSampler),
    ],
)
def test_create_sampler_integer_weights(weights, y_train, expected_need_shuffle, expected_sampler_type):
    y_train = np.array(y_train)
    need_shuffle, sampler = create_sampler(weights, y_train)

    assert need_shuffle == expected_need_shuffle
    assert isinstance(sampler, expected_sampler_type)


@pytest.mark.parametrize(
    "weights, y_train, expected_weights",
    [
        ({0: 0.3, 1: 0.7}, [0, 1, 0, 1], [0.3, 0.7, 0.3, 0.7]),
        ({0: 1.0, 1: 0.5}, [0, 1, 1, 0], [1.0, 0.5, 0.5, 1.0]),
    ],
)
def test_create_sampler_dict_weights(weights, y_train, expected_weights):
    y_train = np.array(y_train)
    expected_weights = np.array(expected_weights)

    need_shuffle, sampler = create_sampler(weights, y_train)

    assert not need_shuffle
    assert isinstance(sampler, WeightedRandomSampler)

    samples_weight = sampler.weights.numpy()
    np.testing.assert_array_almost_equal(samples_weight, expected_weights)


@pytest.mark.parametrize(
    "weights, y_train, expected_weights",
    [
        ([0.2, 0.8, 0.5, 0.5], [0, 1, 0, 1], [0.2, 0.8, 0.5, 0.5]),
        ([1.0, 1.0, 1.0, 1.0], [0, 1, 0, 1], [1.0, 1.0, 1.0, 1.0]),
    ],
)
def test_create_sampler_list_weights(weights, y_train, expected_weights):
    y_train = np.array(y_train)
    weights = np.array(weights)
    expected_weights = np.array(expected_weights)

    need_shuffle, sampler = create_sampler(weights, y_train)

    assert not need_shuffle
    assert isinstance(sampler, WeightedRandomSampler)

    samples_weight = sampler.weights.numpy()
    np.testing.assert_array_almost_equal(samples_weight, expected_weights)


@pytest.mark.parametrize(
    "weights, y_train, expected_error, error_message",
    [
        (
            [0.5, 0.5, 0.5],
            [0, 1],
            ValueError,
            "Custom weights should match number of train samples.",
        ),
        (
            2,
            [0, 1, 0, 1],
            ValueError,
            "Weights should be either 0, 1, dictionnary or list.",
        ),
    ],
)
def test_create_sampler_invalid_inputs(weights, y_train, expected_error, error_message):
    y_train = np.array(y_train)
    with pytest.raises(expected_error, match=error_message):
        create_sampler(weights, y_train)


##############################################


def mock_create_sampler(weights, X_train):
    need_shuffle = weights is not None
    sampler = MagicMock()
    return need_shuffle, sampler


def mock_check_input(X):
    pass


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


@pytest.mark.parametrize(
    "cat_dims, cat_idxs, cat_emb_dim, expected_output",
    [
        ([10, 20], [0, 1], 5, ([10, 20], [0, 1], [5, 5])),
        ([30, 40], [1, 0], [4, 6], ([40, 30], [1, 0], [6, 4])),
    ],
)
def test_check_embedding_parameters_valid(cat_dims, cat_idxs, cat_emb_dim, expected_output):
    assert check_embedding_parameters(cat_dims, cat_idxs, cat_emb_dim) == expected_output


@pytest.mark.parametrize(
    "cat_dims, cat_idxs, cat_emb_dim, error_message",
    [
        (
            [],
            [0, 1],
            5,
            "If cat_idxs is non-empty, cat_dims must be defined as a list of same length.",
        ),
        (
            [10, 20],
            [],
            5,
            "If cat_dims is non-empty, cat_idxs must be defined as a list of same length.",
        ),
        ([10], [0, 1], 5, "The lists cat_dims and cat_idxs must have the same length."),
        # ([10, 20], [0, 1], [5], 'cat_emb_dim and cat_dims must be lists of same length, got 1 and 2'),
    ],
)
def test_check_embedding_parameters_invalid(cat_dims, cat_idxs, cat_emb_dim, error_message):
    with pytest.raises(ValueError, match=error_message):
        check_embedding_parameters(cat_dims, cat_idxs, cat_emb_dim)


@pytest.mark.parametrize(
    "x_train,eval_set",
    [
        (
            np.random.rand(1000, 10),
            [np.random.rand(50, 10), np.random.rand(50, 10), np.random.rand(300, 10)],
        ),
        (
            np.random.rand(10000, 10),
            [np.random.rand(500, 10), np.random.rand(500, 10), np.random.rand(3000, 10)],
        ),
        (
            np.random.rand(10000, 10),
            [],
        ),
    ],
)
@pytest.mark.parametrize(
    "batch_size",
    [128, 1000, 2000, 64, 32, 600],
)
@pytest.mark.parametrize(
    "num_workers,pin_memory",
    [
        (
            0,
            False,
        )
    ],
)
@pytest.mark.parametrize(
    "drop_last",
    [
        True,
        False,
    ],
)
def test_create_dataloaders_pt(
    x_train,
    eval_set,
    batch_size,
    num_workers,
    drop_last,
    pin_memory,
):
    train_dataloader, valid_dataloaders = create_dataloaders_pt(x_train, eval_set, 0, batch_size, num_workers, drop_last, pin_memory)
    assert len(train_dataloader) > 0
    assert len(valid_dataloaders) == len(eval_set)

    assert isinstance(valid_dataloaders, list)
    loaded_data = [d for d in train_dataloader]
    ceil_div = math.ceil(len(x_train) / batch_size)
    assert len(loaded_data) == len(train_dataloader)
    assert len(loaded_data) == ceil_div if not drop_last else True
    assert len(loaded_data) == ceil_div - 1 if drop_last and ceil_div > 1 else True
    for i, vda in enumerate(valid_dataloaders):
        v_loaded_data = [v[0] for v in vda]
        cat_data = torch.cat(v_loaded_data)
        assert len(cat_data) == len(vda.dataset.x)
        assert torch.equal(cat_data, vda.dataset.x)


@pytest.mark.parametrize(
    "x_train,y_train,eval_set",
    [
        (
            np.random.rand(1000, 10),
            np.random.rand(1000, 1),
            [
                (np.random.rand(50, 10), np.random.rand(50, 1)),
                (np.random.rand(50, 10), np.random.rand(50, 1)),
                (np.random.rand(300, 10), np.random.rand(300, 1)),
            ],
        ),
        (
            np.random.rand(10000, 10),
            np.random.rand(10000, 1),
            [
                (np.random.rand(500, 10), np.random.rand(500, 1)),
                (np.random.rand(500, 10), np.random.rand(500, 1)),
                (np.random.rand(3000, 10), np.random.rand(3000, 1)),
            ],
        ),
        (
            np.random.rand(10000, 10),
            np.random.rand(10000, 1),
            [],
        ),
    ],
)
@pytest.mark.parametrize(
    "batch_size",
    [128, 1000, 2000, 64, 32, 600],
)
@pytest.mark.parametrize(
    "num_workers,pin_memory",
    [
        (
            0,
            False,
        )
    ],
)
@pytest.mark.parametrize(
    "drop_last",
    [
        True,
        False,
    ],
)
# @pytest.mark.skip("flaky")
def test_create_dataloaders(
    x_train,
    y_train,
    eval_set,
    batch_size,
    num_workers,
    drop_last,
    pin_memory,
):
    train_dataloader, valid_dataloaders = create_dataloaders(x_train, y_train, eval_set, 0, batch_size, num_workers, drop_last, pin_memory)
    assert len(train_dataloader) > 0
    assert len(valid_dataloaders) == len(eval_set)

    assert isinstance(valid_dataloaders, list)
    loaded_data = [d for d in train_dataloader]
    ceil_div = math.ceil(len(x_train) / batch_size)
    assert len(loaded_data) == len(train_dataloader)
    assert len(loaded_data) == ceil_div if not drop_last else True
    assert len(loaded_data) == ceil_div - 1 if drop_last and ceil_div > 1 else True
    for i, vda in enumerate(valid_dataloaders):
        v_loaded_data = [v[0] for v in vda]
        cat_data = torch.cat(v_loaded_data)
        assert len(cat_data) == len(vda.dataset.x)
        assert torch.equal(cat_data, vda.dataset.x)
