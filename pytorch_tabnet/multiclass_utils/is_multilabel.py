from typing import Union

import numpy as np
from scipy.sparse import dok_matrix, issparse, lil_matrix
from scipy.sparse.base import spmatrix

from pytorch_tabnet.multiclass_utils._is_integral_float import _is_integral_float


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
    if hasattr(y, "__array__"):
        y = np.asarray(y)
    if not (hasattr(y, "shape") and y.ndim == 2 and y.shape[1] > 1):
        return False

    if issparse(y):
        if isinstance(y, (dok_matrix, lil_matrix)):
            y = y.tocsr()
        return (
            len(y.data) == 0
            or np.unique(y.data).size == 1
            and (
                y.dtype.kind in "biu" or _is_integral_float(np.unique(y.data))  # bool, int, uint
            )
        )
    else:
        labels = np.unique(y)

        return len(labels) < 3 and (
            y.dtype.kind in "biu" or _is_integral_float(labels)  # bool, int, uint
        )
