import numpy as np
import pytest
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler

from pytorch_tabnet.data_handlers import create_dataloaders, create_sampler


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

    assert isinstance(train_loader, DataLoader)
    assert len(train_loader) == len(y_train) // batch_size
    assert len(valid_loaders) == len(eval_set)
    for val_loader in valid_loaders:
        assert isinstance(val_loader, DataLoader)


def test_create_dataloaders_sparse_data(sparse_sample_data):
    X_train, y_train, eval_set = sparse_sample_data
    batch_size = 16
    num_workers = 0
    drop_last = False
    pin_memory = False
    weights = 0

    train_loader, valid_loaders = create_dataloaders(X_train, y_train, eval_set, weights, batch_size, num_workers, drop_last, pin_memory)

    assert isinstance(train_loader, DataLoader)
    assert len(train_loader) == (len(y_train) + batch_size - 1) // batch_size
    assert len(valid_loaders) == len(eval_set)
    for val_loader in valid_loaders:
        assert isinstance(val_loader, DataLoader)


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
