"""Validation utility functions for TabNet."""

from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.utils import check_array

from ..utils.label_processing import unique_labels
from ._assert_all_finite import _assert_all_finite


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
    eval_set: Union[List[Tuple[np.ndarray, np.ndarray]], List[np.ndarray]],
    eval_name: Optional[List[str]],
    X_train: np.ndarray,
    y_train: Optional[np.ndarray] = None,
) -> None:
    """Check if the shapes of eval_set are compatible with (X_train, y_train).

    Parameters
    ----------
    eval_set : list of tuple or list of arrays
        For supervised learning: List of eval tuple set (X, y).
        For unsupervised learning: List of eval arrays (X only).
        The last one is used for early stopping
    eval_name : list of str
        List of eval set names.
    X_train : np.ndarray
        Train owned products
    y_train : np.array, optional
        Train targeted products. If None, only X validation is performed (for pretraining).



    """
    assert len(eval_set) == len(eval_name), "eval_set and eval_name have not the same length"

    # Determine if this is supervised (with y) or unsupervised (X only) validation
    is_supervised = y_train is not None

    if len(eval_set) > 0:
        if is_supervised:
            assert all(len(elem) == 2 for elem in eval_set), "Each tuple of eval_set need to have two elements"
        else:
            # For unsupervised, eval_set should be List[np.ndarray]
            assert all(isinstance(elem, np.ndarray) for elem in eval_set), (
                "For unsupervised pretraining, eval_set should be a list of arrays"
            )

    for i, eval_item in enumerate(eval_set):
        name = eval_name[i]

        if is_supervised:
            # Supervised case: eval_item is (X, y) tuple
            X, y = eval_item
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
        else:
            # Unsupervised case: eval_item is just X array
            X = eval_item
            check_array(X)

            msg = f"Number of columns is different between eval set {i}" + f"({X.shape[1]}) and X_train ({X_train.shape[1]})"
            assert X.shape[1] == X_train.shape[1], msg


def _get_sparse_data(X: np.ndarray) -> np.ndarray:
    return X


def assert_all_finite(X: np.ndarray, allow_nan: bool = False) -> None:
    """Throw a ValueError if X contains NaN or infinity.

    Parameters
    ----------
    X : np.ndarray
        Array to check for NaN or infinity values.
    allow_nan : bool, default=False
        Whether to allow NaN values.

    Raises
    ------
    ValueError
        If X contains NaN or infinity values.

    """  # noqa: DOC502
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


def check_list_groups(list_groups: List[List[int]], input_dim: int) -> None:
    """Check that list_groups is valid for group matrix construction.

    - Is a list of list
    - Does not contain the same feature in different groups
    - Does not contain unknown features (>= input_dim)
    - Does not contain empty groups.

    Parameters
    ----------
    list_groups : list of list of int
        Each element is a list representing features in the same group.
        One feature should appear in maximum one group.
        Feature that don't get assigned a group will be in their own group of one feature.
    input_dim : int
        Number of features in the initial dataset

    """
    assert isinstance(list_groups, list), "list_groups must be a list of list."

    if len(list_groups) == 0:
        return
    else:
        for group_pos, group in enumerate(list_groups):
            msg = f"Groups must be given as a list of list, but found {group} in position {group_pos}."  # noqa
            assert isinstance(group, list), msg
            assert len(group) > 0, "Empty groups are forbidding please remove empty groups []"

    n_elements_in_groups = np.sum([len(group) for group in list_groups])
    flat_list = []
    for group in list_groups:
        flat_list.extend(group)
    unique_elements = np.unique(flat_list)
    n_unique_elements_in_groups = len(unique_elements)
    msg = "One feature can only appear in one group, please check your grouped_features."
    assert n_unique_elements_in_groups == n_elements_in_groups, msg

    highest_feat = np.max(unique_elements)
    assert highest_feat < input_dim, f"Number of features is {input_dim} but one group contains {highest_feat}."  # noqa
    return


def _validate_input(y: np.ndarray) -> None:
    """Validate that input is array-like and not a string.

    Parameters
    ----------
    y : np.ndarray
        Input to validate.

    Raises
    ------
    ValueError
        If input is not array-like or is a string.

    """
    from typing import Sequence

    if not ((isinstance(y, (Sequence)) or hasattr(y, "__array__")) and not isinstance(y, str)):
        raise ValueError("Expected array-like (array or non-string sequence), got %r" % y)


def _validate_multitask_shape(y_train: np.ndarray) -> None:
    """Validate that y_train has the correct shape for multitask learning.

    Parameters
    ----------
    y_train : np.ndarray
        Training targets array.

    Raises
    ------
    ValueError
        If y_train doesn't have at least 2 dimensions.

    """
    if len(y_train.shape) < 2:
        raise ValueError("y_train should be of shape (n_examples, n_tasks)" + f"but got {y_train.shape}")
