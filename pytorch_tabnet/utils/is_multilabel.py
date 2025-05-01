"""Utilities for checking multiclass labels in TabNet."""

import numpy as np

from pytorch_tabnet.utils._is_integral_float import _is_integral_float


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
    """Check if ``y`` is in a multilabel format.

    Parameters
    ----------
    y : numpy array of shape [n_samples]
        Target values.

    Returns
    -------
    out : bool
        Return ``True``, if ``y`` is in a multilabel format, else ```False``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils.multiclass import is_multilabel
    >>> is_multilabel([0, 1, 0, 1])
    False
    >>> is_multilabel([[1], [0, 2], []])
    False
    >>> is_multilabel(np.array([[1, 0], [0, 0]]))
    True
    >>> is_multilabel(np.array([[1], [0], [0]]))
    False
    >>> is_multilabel(np.array([[1, 0, 0]]))
    True

    """
    if _has_array_like_properties(y):
        y = np.asarray(y)
    if not _has_required_shape(y):
        return False

    return _is_valid_dense_multilabel(y)
