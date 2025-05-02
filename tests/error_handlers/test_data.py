import numpy as np
import pytest

from pytorch_tabnet.error_handlers.data import (
    check_data_general,
    model_input_and_target_data_check,
    model_input_data_check,
    model_target_check,  # Import the private function for testing
)


# Fixtures for valid data
@pytest.fixture()
def valid_numpy_int_array():
    """Return a valid numpy int array."""
    return np.array([[1, 2], [3, 4]])


@pytest.fixture()
def valid_numpy_float_array():
    """Return a valid numpy float array."""
    return np.array([[1.0, 2.0], [3.0, 4.0]])


@pytest.fixture()
def valid_numpy_bool_array():
    """Return a valid numpy bool array."""
    return np.array([[True, False], [False, True]])


@pytest.fixture()
def valid_target_array():
    """Return a valid target numpy array."""
    return np.array([0, 1])


# Tests for check_data_general
def test_check_data_general_valid(valid_numpy_int_array, valid_numpy_float_array, valid_numpy_bool_array):
    """Test check_data_general with valid inputs."""
    check_data_general(valid_numpy_int_array)
    check_data_general(valid_numpy_float_array)
    check_data_general(valid_numpy_bool_array)


def test_check_data_general_not_numpy():
    """Test check_data_general with non-numpy array input."""
    with pytest.raises(TypeError, match="Input data must be a numpy array."):
        check_data_general([1, 2, 3])


def test_check_data_general_unsupported_dtype():
    """Test check_data_general with unsupported dtype."""
    data = np.array(["a", "b"])
    # Match the specific error message format including the dtype
    match_str = f"Data type {data.dtype} not supported. Allowed types: float, int, bool."
    with pytest.raises(TypeError, match=match_str):
        check_data_general(data)


def test_check_data_general_empty_array():
    """Test check_data_general with an empty array."""
    data = np.array([])
    with pytest.raises(ValueError, match="Input data cannot be empty."):
        check_data_general(data)


# Tests for model_input_data_check
def test_model_input_data_check_valid(valid_numpy_float_array):
    """Test model_input_data_check with valid input."""
    model_input_data_check(valid_numpy_float_array)


def test_model_input_data_check_not_2d():
    """Test model_input_data_check with non-2D array."""
    data = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="Input data must be 2D, but got 1 dimensions."):
        model_input_data_check(data)


def test_model_input_data_check_contains_nan():
    """Test model_input_data_check with NaN values."""
    data = np.array([[1.0, 2.0], [np.nan, 4.0]])
    with pytest.raises(ValueError, match="Input data contains NaN values."):
        model_input_data_check(data)


def test_model_input_data_check_contains_inf():
    """Test model_input_data_check with infinite values."""
    data = np.array([[1.0, 2.0], [np.inf, 4.0]])
    with pytest.raises(ValueError, match="Input data contains infinite values."):
        model_input_data_check(data)


# Tests for _model_target_check
def test__model_target_check_valid(valid_target_array):
    """Test _model_target_check with valid 1D input."""
    model_target_check(valid_target_array)


def test__model_target_check_valid_2d():
    """Test _model_target_check with valid 2D input."""
    target = np.array([[0], [1]])
    model_target_check(target)


def test__model_target_check_invalid_dim():
    """Test _model_target_check with invalid dimensions."""
    target = np.array([[[0]], [[1]]])  # 3D array
    with pytest.raises(ValueError, match="Input target must be 1D or 2D, but got 3 dimensions."):
        model_target_check(target)


def test__model_target_check_contains_nan():
    """Test _model_target_check with NaN values."""
    target = np.array([0, np.nan])
    with pytest.raises(ValueError, match="Input target contains NaN values."):
        model_target_check(target)


def test__model_target_check_contains_inf():
    """Test _model_target_check with infinite values."""
    target = np.array([0, np.inf])
    with pytest.raises(ValueError, match="Input target contains infinite values."):
        model_target_check(target)


# Tests for model_input_and_target_data_check
def test_model_input_and_target_data_check_valid(valid_numpy_float_array, valid_target_array):
    """Test model_input_and_target_data_check with valid inputs."""
    model_input_and_target_data_check(valid_numpy_float_array, valid_target_array)


def test_model_input_and_target_data_check_mismatch_samples(valid_numpy_float_array):
    """Test model_input_and_target_data_check with mismatched sample sizes."""
    target = np.array([0, 1, 2])  # Has 3 samples, data has 2
    match_str = rf"Number of samples in data \({valid_numpy_float_array.shape[0]}\) does not match target \({target.shape[0]}\)."
    with pytest.raises(ValueError, match=match_str):
        model_input_and_target_data_check(valid_numpy_float_array, target)


def test_model_input_and_target_data_check_invalid_data(valid_target_array):
    """Test model_input_and_target_data_check with invalid data input."""
    data = np.array([[1.0, 2.0], [np.nan, 4.0]])  # Contains NaN
    with pytest.raises(ValueError, match="Input data contains NaN values."):
        model_input_and_target_data_check(data, valid_target_array)


def test_model_input_and_target_data_check_invalid_target(valid_numpy_float_array):
    """Test model_input_and_target_data_check with invalid target input."""
    target = np.array([])  # Empty target
    with pytest.raises(ValueError, match="Input data cannot be empty."):
        model_input_and_target_data_check(valid_numpy_float_array, target)

    target_nan = np.array([0, np.nan])  # Target with NaN
    with pytest.raises(ValueError, match="Input target contains NaN values."):
        model_input_and_target_data_check(valid_numpy_float_array, target_nan)
