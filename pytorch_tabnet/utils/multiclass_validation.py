"""Validation utilities for multiclass classification in TabNet."""

from typing import Union

import numpy as np
import pandas as pd
from scipy import sparse as sp  # todo: replace scipy with numpy
from scipy.sparse.base import spmatrix  # todo: replace scipy with numpy

from pytorch_tabnet.utils._assert_all_finite import _assert_all_finite
from pytorch_tabnet.utils.label_processing import unique_labels


def _get_sparse_data(X: Union[np.ndarray, spmatrix]) -> Union[np.ndarray, spmatrix]:
    return X.data if sp.issparse(X) else X


def assert_all_finite(X: Union[np.ndarray, spmatrix], allow_nan: bool = False) -> None:
    """Throw a ValueError if X contains NaN or infinity.

    Parameters
    ----------
    X : array or sparse matrix
    allow_nan : bool

    """
    _assert_all_finite(_get_sparse_data(X), allow_nan)


def _get_target_types(y: np.ndarray) -> np.ndarray:
    return pd.Series(y).map(type).unique()


def _has_consistent_types(types: np.ndarray) -> bool:
    return len(types) == 1


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
    target_types = _get_target_types(y)
    if not _has_consistent_types(target_types):
        raise TypeError(f"Values on the target must have the same type. Target has types {target_types}")


def _are_all_labels_valid(valid_labels: np.ndarray, labels: np.ndarray) -> bool:
    return set(valid_labels).issubset(set(labels))


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
        if not _are_all_labels_valid(valid_labels, labels):
            raise ValueError(
                f"""Valid set -- {set(valid_labels)} --\n" +
                "contains unkown targets from training --\n" +
                f"{set(labels)}"""
            )
    return
