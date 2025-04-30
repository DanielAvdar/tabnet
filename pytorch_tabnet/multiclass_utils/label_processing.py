"""Label processing utilities for multiclass classification in TabNet."""

from itertools import chain
from typing import List

import numpy as np

from .type_of_target import type_of_target


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
    if not ys:
        raise ValueError("No argument has been passed.")
    # Check that we don't mix label format

    ys_types = set(type_of_target(x) for x in ys)
    if ys_types == {"binary", "multiclass"}:
        ys_types = {"multiclass"}

    if len(ys_types) > 1:
        raise ValueError("Mix type of y not allowed, got types %s" % ys_types)

    label_type = ys_types.pop()

    # Get the unique set of labels
    _unique_labels = _FN_UNIQUE_LABELS.get(label_type, None)
    if not _unique_labels:
        raise ValueError("Unknown label type: %s" % repr(ys))

    ys_labels = set(chain.from_iterable(_unique_labels(y) for y in ys))

    # Check that we don't mix string type with number type
    if len(set(isinstance(label, str) for label in ys_labels)) > 1:
        raise ValueError("Mix of label input types (string and number)")

    return np.array(sorted(ys_labels))
