"""Validation utilities for multiclass classification in TabNet."""

from typing import Union

import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.sparse.base import spmatrix

from ._assert_all_finite import _assert_all_finite
from .label_processing import unique_labels

# def _assert_all_finite(X: np.ndarray, allow_nan: bool = False) -> None:


# def assert_all_finite(X: spmatrix, allow_nan: bool = False) -> None:


# def check_classification_targets(y: np.ndarray) -> None:


# def check_unique_type(y: np.ndarray) -> None:


# def check_output_dim(labels: np.ndarray, y: np.ndarray) -> None:


def assert_all_finite(X: Union[np.ndarray, spmatrix], allow_nan: bool = False) -> None:
    """Throw a ValueError if X contains NaN or infinity.

    Parameters
    ----------
    X : array or sparse matrix
    allow_nan : bool

    """
    _assert_all_finite(X.data if sp.issparse(X) else X, allow_nan)


def check_unique_type(y: np.ndarray) -> None:
    """Check that all elements in y have the same type.

    Parameters
    ----------
    y : np.ndarray
        Target array to check.

    Raises
    ------
    TypeError
        If values in y have different types.

    """
    target_types = pd.Series(y).map(type).unique()
    if len(target_types) != 1:
        raise TypeError(f"Values on the target must have the same type. Target has types {target_types}")


def check_output_dim(labels: np.ndarray, y: np.ndarray) -> None:
    """Check that all labels in y are present in the training labels.

    Parameters
    ----------
    labels : np.ndarray
        Array of valid labels from training.
    y : np.ndarray
        Array of labels to check.

    Raises
    ------
    ValueError
        If y contains labels not present in labels.

    """
    if y is not None:
        check_unique_type(y)
        valid_labels = unique_labels(y)
        if not set(valid_labels).issubset(set(labels)):
            raise ValueError(
                f"""Valid set -- {set(valid_labels)} --\n" +
                "contains unkown targets from training --\n" +
                f"{set(labels)}"""
            )
    return
