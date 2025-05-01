"""Label processing utilities for TabNet."""

from itertools import chain
from typing import Callable, List, Set

import numpy as np

from pytorch_tabnet.utils.type_of_target import type_of_target


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

    Examples
    --------
    >>> from sklearn.utils.multiclass import unique_labels
    >>> unique_labels([3, 5, 5, 5, 7, 7])
    array([3, 5, 7])
    >>> unique_labels([1, 2, 3, 4], [2, 2, 3, 4])
    array([1, 2, 3, 4])
    >>> unique_labels([1, 2, 10], [5, 11])
    array([ 1,  2,  5, 10, 11])

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
