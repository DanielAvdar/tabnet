"""Validation utility functions for TabNet."""

from typing import Callable, List, Optional, Set, Tuple, Union
from typing import Callable, List, Optional, Set, Sequence, Tuple, Union
from itertools import chain

import numpy as np
from sklearn.utils import check_array

# Include _assert_all_finite functionality directly to avoid circular imports
def _is_float_dtype(X: np.ndarray) -> bool:
    return X.dtype.kind in "fc"


def _has_finite_sum(X: np.ndarray) -> bool:
    return np.isfinite(np.sum(X))


def _check_float_array(X: np.ndarray, allow_nan: bool) -> None:
    msg_err = "Input contains {} or a value too large for {!r}."
    has_inf = np.isinf(X).any()
    has_nan = not np.isfinite(X).all()

    if (allow_nan and has_inf) or (not allow_nan and has_nan):
        type_err = "infinity" if allow_nan else "NaN, infinity"
        raise ValueError(msg_err.format(type_err, X.dtype))


def _check_object_array_for_nan(X: np.ndarray, allow_nan: bool) -> None:
    if not allow_nan and np.isnan(X).any():
        raise ValueError("Input contains NaN")


def _assert_all_finite(X: np.ndarray, allow_nan: bool = False) -> None:
    """Like assert_all_finite, but only for ndarray."""
    X = np.asanyarray(X)

    is_float = _is_float_dtype(X)

    if is_float and _has_finite_sum(X):
        pass
    elif is_float:
        _check_float_array(X, allow_nan)
    # for object dtype data, we only check for NaNs (GH-13254)
    elif X.dtype == np.dtype("object") and not allow_nan:
        _check_object_array_for_nan(X, allow_nan)


# Support functions for is_multilabel
def _is_integral_float(y: np.ndarray) -> bool:
    return y.dtype.kind == "f" and np.all(y.astype(int) == y)


def _has_array_like_properties(y: np.ndarray) -> bool:
    return hasattr(y, "__array__")


def _has_required_shape(y: np.ndarray) -> bool:
    return hasattr(y, "shape") and y.ndim == 2 and y.shape[1] > 1


def _is_valid_dense_multilabel(y: np.ndarray) -> bool:
    labels = np.unique(y)
    return len(labels) < 3 and (
        y.dtype.kind in "biu" or _is_integral_float(labels)  # bool, int, uint
    )


def is_multilabel(y: np.ndarray) -> bool:
    """Check if ``y`` is in a multilabel format."""
    if _has_array_like_properties(y):
        y = np.asarray(y)
    if not _has_required_shape(y):
        return False

    return _is_valid_dense_multilabel(y)


# Support functions for type_of_target
def _is_valid_input_type(y: np.ndarray) -> bool:
    return (isinstance(y, (Sequence)) or hasattr(y, "__array__")) and not isinstance(y, str)


def _is_sparse_series(y: np.ndarray) -> bool:
    return y.__class__.__name__ == "SparseSeries"


def _is_invalid_dimension(y: np.ndarray) -> bool:
    """Check if y has invalid dimensions."""
    return bool(y.ndim > 2 or (y.dtype == object and len(y) and not isinstance(y.flat[0], str)))


def _is_empty_2d_array(y: np.ndarray) -> bool:
    return y.ndim == 2 and y.shape[1] == 0


def _get_multioutput_suffix(y: np.ndarray) -> str:
    if y.ndim == 2 and y.shape[1] > 1:
        return "-multioutput"  # [[1, 2], [1, 2]]
    else:
        return ""  # [1, 2, 3] or [[1], [2], [3]]


def _is_continuous_float(y: np.ndarray) -> bool:
    return y.dtype.kind == "f" and np.any(y != y.astype(int))


def _is_multiclass(y: np.ndarray) -> bool:
    """Check if y contains more than two discrete values."""
    return bool((len(np.unique(y)) > 2) or (y.ndim >= 2 and len(y[0]) > 1))


def _validate_input(y: np.ndarray) -> None:
    if not _is_valid_input_type(y):
        raise ValueError("Expected array-like (array or non-string sequence), got %r" % y)


def type_of_target(y: np.ndarray) -> str:
    """Determine the type of data indicated by the target."""
    _validate_input(y)

    if is_multilabel(y):
        return "multilabel-indicator"

    try:
        y = np.asarray(y)
    except ValueError:
        # Known to fail in numpy 1.3 for array of arrays
        return "unknown"

    # Invalid inputs
    if _is_invalid_dimension(y):
        return "unknown"  # [[[1, 2]]] or [obj_1] and not ["label_1"]

    if _is_empty_2d_array(y):
        return "unknown"  # [[]]

    suffix = _get_multioutput_suffix(y)

    # check float and contains non-integer float values
    if _is_continuous_float(y):
        # [.1, .2, 3] or [[.1, .2, 3]] or [[1., .2]] and not [1., 2., 3.]
        _assert_all_finite(y)
        return "continuous" + suffix

    if _is_multiclass(y):
        return "multiclass" + suffix  # [1, 2, 3] or [[1., 2., 3]] or [[1, 2]]
    else:
        return "binary"  # [1, 2] or [["a"], ["b"]]


# Include unique_labels functionality directly to avoid circular imports
def _unique_multiclass(y: np.ndarray) -> np.ndarray:
    if hasattr(y, "__array__"):
        return np.unique(np.asarray(y))
    else:
        return np.array(list(set(y)))


def _unique_indicator(y: np.ndarray) -> np.ndarray:
    """Not implemented."""
    raise IndexError(
        f"""Given labels are of size {y.shape} while they should be (n_samples,) \n"""
        + """If attempting multilabel classification, try using TabNetMultiTaskClassification """
        + """or TabNetRegressor"""
    )


_FN_UNIQUE_LABELS = {
    "binary": _unique_multiclass,
    "multiclass": _unique_multiclass,
    "multilabel-indicator": _unique_indicator,
}


def _check_no_arguments(ys: List[np.ndarray]) -> None:
    if not ys:
        raise ValueError("No argument has been passed.")


def _consolidate_label_types(ys_types: Set[str]) -> Set[str]:
    if ys_types == {"binary", "multiclass"}:
        return {"multiclass"}
    return ys_types


def _validate_label_types(ys_types: Set[str]) -> str:
    if len(ys_types) > 1:
        raise ValueError("Mix type of y not allowed, got types %s" % ys_types)
    return ys_types.pop()


def _get_unique_labels_function(label_type: str) -> Callable[[np.ndarray], np.ndarray]:
    _unique_labels = _FN_UNIQUE_LABELS.get(label_type, None)
    if not _unique_labels:
        raise ValueError("Unknown label type: %s" % repr(label_type))
    return _unique_labels


def _extract_all_labels(ys: List[np.ndarray], unique_labels_fn: Callable[[np.ndarray], np.ndarray]) -> Set:
    return set(chain.from_iterable(unique_labels_fn(y) for y in ys))


def _validate_label_input_types(ys_labels: Set) -> None:
    if len(set(isinstance(label, str) for label in ys_labels)) > 1:
        raise ValueError("Mix of label input types (string and number)")


def unique_labels(*ys: List[np.ndarray]) -> np.ndarray:
    """Extract an ordered array of unique labels.

    We don't allow:
        - mix of multilabel and multiclass (single label) targets
        - mix of label indicator matrix and anything else,
          because there are no explicit labels)
        - mix of label indicator matrices of different sizes
        - mix of string and integer labels

    At the moment, we also don't allow "multiclass-multioutput" input type.

    Parameters
    ----------
    *ys : array-likes

    Returns
    -------
    out : numpy array of shape [n_unique_labels]
        An ordered array of unique labels.
    """
    _check_no_arguments(list(ys))

    # Check that we don't mix label format
    ys_types = set(type_of_target(x) for x in ys)
    ys_types = _consolidate_label_types(ys_types)

    label_type = _validate_label_types(ys_types)

    # Get the unique set of labels
    unique_labels_fn = _get_unique_labels_function(label_type)

    ys_labels = _extract_all_labels(list(ys), unique_labels_fn)

    # Check that we don't mix string type with number type
    _validate_label_input_types(ys_labels)

    return np.array(sorted(ys_labels))


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
