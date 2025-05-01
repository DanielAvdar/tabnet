"""Utilities for checking multiclass labels in TabNet."""

from typing import Union

import numpy as np
from scipy.sparse import dok_matrix, issparse, lil_matrix
from scipy.sparse.base import spmatrix

from pytorch_tabnet.utils._is_integral_float import _is_integral_float


def _has_array_like_properties(y: Union[np.ndarray, spmatrix]) -> bool:
    return hasattr(y, "__array__")


def _has_required_shape(y: Union[np.ndarray, spmatrix]) -> bool:
    return hasattr(y, "shape") and y.ndim == 2 and y.shape[1] > 1


def _convert_sparse_to_csr(y: spmatrix) -> spmatrix:
    if isinstance(y, (dok_matrix, lil_matrix)):
        return y.tocsr()
    return y


def _is_valid_sparse_multilabel(y: spmatrix) -> bool:
    unique_data = np.unique(y.data)
    return (
        len(y.data) == 0
        or unique_data.size == 1
        and (
            y.dtype.kind in "biu" or _is_integral_float(unique_data)  # bool, int, uint
        )
    )


def _is_valid_dense_multilabel(y: np.ndarray) -> bool:
    labels = np.unique(y)
    return len(labels) < 3 and (
        y.dtype.kind in "biu" or _is_integral_float(labels)  # bool, int, uint
    )


def is_multilabel(y: Union[np.ndarray, spmatrix]) -> bool:
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

    if issparse(y):
        y = _convert_sparse_to_csr(y)
        return _is_valid_sparse_multilabel(y)
    else:
        return _is_valid_dense_multilabel(y)
