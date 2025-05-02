"""Validation utilities for multiclass classification in TabNet."""

import numpy as np

from pytorch_tabnet.utils._assert_all_finite import _assert_all_finite
from pytorch_tabnet.utils.label_processing import unique_labels


def _get_sparse_data(X: np.ndarray) -> np.ndarray:
    return X


def assert_all_finite(X: np.ndarray, allow_nan: bool = False) -> None:
    """Throw a ValueError if X contains NaN or infinity.

    Parameters
    ----------
    X : array or sparse matrix
    allow_nan : bool

    """
    _assert_all_finite(_get_sparse_data(X), allow_nan)


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
        valid_labels = unique_labels(y)
        if not _are_all_labels_valid(valid_labels, labels):
            raise ValueError(
                f"""Valid set -- {set(valid_labels)} --\n" +
                "contains unkown targets from training --\n" +
                f"{set(labels)}"""
            )
    return
