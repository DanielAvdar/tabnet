"""Data validation functions."""

from typing import Any

import numpy as np


def check_data_general(
    data: Any,
) -> None:
    """Check data format and values.

    - 1: data should be a numpy array
    - 2: data type should be only of the following: float, int, bool or there variants
    - 3: data should not be empty.

    Parameters
    ----------
    data : Any
        Object to check

    Raises
    ------
    TypeError
        If data is not a numpy array or if dtype is not supported.
    ValueError
        If data is empty.

    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")

    allowed_types = [np.floating, np.integer, np.bool_]
    if not any(np.issubdtype(data.dtype, t) for t in allowed_types):
        raise TypeError(f"Data type {data.dtype} not supported. Allowed types: float, int, bool.")

    if data.size == 0:
        raise ValueError("Input data cannot be empty.")


def model_input_data_check(data: Any) -> None:
    """Check data format and values.

    - 1: data should be check_data_general compatible
    - 2: shape should be 2D
    - 3: data should not contain NaN values
    - 4: data should not contain infinite values.

    Parameters
    ----------
    data : Any
        Object to check

    Raises
    ------
    ValueError
        If data is not 2D, contains NaN, or contains infinite values.

    """
    check_data_general(data)
    if data.ndim != 2:
        raise ValueError(f"Input data must be 2D, but got {data.ndim} dimensions.")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values.")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values.")


def model_target_check(target: Any) -> None:
    """Check target format and values.

    - 1: target should be check_data_general compatible
    - 2: shape should be 1D or 2D
    - 3: target should not contain NaN values
    - 4: target should not contain infinite values.

    Parameters
    ----------
    target : Any
        Object to check

    Raises
    ------
    ValueError
        If target is not 1D or 2D, contains NaN, or contains infinite values.

    """
    check_data_general(target)
    if target.ndim not in [1, 2]:
        raise ValueError(f"Input target must be 1D or 2D, but got {target.ndim} dimensions.")
    if np.isnan(target).any():
        raise ValueError("Input target contains NaN values.")
    if np.isinf(target).any():
        raise ValueError("Input target contains infinite values.")


def model_input_and_target_data_check(data: Any, target: Any) -> None:
    """Check data format and values.

    - 2: target should be _model_target_check compatible
    - 3: data should be model_input_data_check compatible
    - 4: data shape[0] should be equal to target shape[0].

    Parameters
    ----------
    data : Any
        Object to check
    target : Any
        Object to check

    Raises
    ------
    ValueError
        If data and target number of samples do not match.

    """
    model_input_data_check(data)
    model_target_check(target)
    if data.shape[0] != target.shape[0]:
        raise ValueError(f"Number of samples in data ({data.shape[0]}) does not match target ({target.shape[0]}).")
