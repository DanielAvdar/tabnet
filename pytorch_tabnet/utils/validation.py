"""Validation utility functions for TabNet."""

from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.utils import check_array


def filter_weights(weights: Union[int, List, np.ndarray]) -> None:
    """Ensure weights are in correct format for regression and multitask TabNet.

    Parameters
    ----------
    weights : int, dict or list
        Initial weights parameters given by user

    Raises
    ------
    ValueError
        If weights are not in the correct format for regression, multitask, or pretraining.

    """
    err_msg = """Please provide a list or np.array of weights for """
    err_msg += """regression, multitask or pretraining: """
    if isinstance(weights, int):
        if weights == 1:
            raise ValueError(err_msg + "1 given.")
    if isinstance(weights, dict):
        raise ValueError(err_msg + "Dict given.")
    return


def validate_eval_set(
    eval_set: List[Tuple[np.ndarray, np.ndarray]],
    eval_name: Optional[List[str]],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> None:
    """Check if the shapes of eval_set are compatible with (X_train, y_train).

    Parameters
    ----------
    eval_set : list of tuple
        List of eval tuple set (X, y).
        The last one is used for early stopping
    eval_name : list of str
        List of eval set names.
    X_train : np.ndarray
        Train owned products
    y_train : np.array
        Train targeted products

    Returns
    -------
    eval_names : list of str
        Validated list of eval_names.
    eval_set : list of tuple
        Validated list of eval_set.

    """

    assert len(eval_set) == len(eval_name), "eval_set and eval_name have not the same length"
    if len(eval_set) > 0:
        assert all(len(elem) == 2 for elem in eval_set), "Each tuple of eval_set need to have two elements"
    for name, (X, y) in zip(eval_name, eval_set, strict=False):
        check_array(X)
        msg = f"Dimension mismatch between X_{name} " + f"{X.shape} and X_train {X_train.shape}"
        assert len(X.shape) == len(X_train.shape), msg

        msg = f"Dimension mismatch between y_{name} " + f"{y.shape} and y_train {y_train.shape}"
        assert len(y.shape) == len(y_train.shape), msg

        msg = f"Number of columns is different between X_{name} " + f"({X.shape[1]}) and X_train ({X_train.shape[1]})"
        assert X.shape[1] == X_train.shape[1], msg

        if len(y_train.shape) == 2:
            msg = f"Number of columns is different between y_{name} " + f"({y.shape[1]}) and y_train ({y_train.shape[1]})"
            assert y.shape[1] == y_train.shape[1], msg
        msg = f"You need the same number of rows between X_{name} " + f"({X.shape[0]}) and y_{name} ({y.shape[0]})"
        assert X.shape[0] == y.shape[0], msg
