"""Target type detection utilities for TabNet."""

from typing import Sequence

import numpy as np

from pytorch_tabnet.utils._assert_all_finite import _assert_all_finite
from pytorch_tabnet.utils.is_multilabel import is_multilabel


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
    """Determine the type of data indicated by the target.

    Note that this type is the most specific type that can be inferred.
    For example:

        * ``binary`` is more specific but compatible with ``multiclass``.
        * ``multiclass`` of integers is more specific but compatible with
          ``continuous``.
        * ``multilabel-indicator`` is more specific but compatible with
          ``multiclass-multioutput``.

    Parameters
    ----------
    y : array-like

    Returns
    -------
    target_type : string
        One of:

        * 'continuous': `y` is an array-like of floats that are not all
          integers, and is 1d or a column vector.
        * 'continuous-multioutput': `y` is a 2d array of floats that are
          not all integers, and both dimensions are of size > 1.
        * 'binary': `y` contains <= 2 discrete values and is 1d or a column
          vector.
        * 'multiclass': `y` contains more than two discrete values, is not a
          sequence of sequences, and is 1d or a column vector.
        * 'multiclass-multioutput': `y` is a 2d array that contains more
          than two discrete values, is not a sequence of sequences, and both
          dimensions are of size > 1.
        * 'multilabel-indicator': `y` is a label indicator matrix, an array
          of two dimensions with at least two columns, and at most 2 unique
          values.
        * 'unknown': `y` is array-like but none of the above, such as a 3d
          array, sequence of sequences, or an array of non-sequence objects.

    Examples
    --------
    >>> import numpy as np
    >>> type_of_target([0.1, 0.6])
    'continuous'
    >>> type_of_target([1, -1, -1, 1])
    'binary'
    >>> type_of_target(['a', 'b', 'a'])
    'binary'
    >>> type_of_target([1.0, 2.0])
    'binary'
    >>> type_of_target([1, 0, 2])
    'multiclass'
    >>> type_of_target([1.0, 0.0, 3.0])
    'multiclass'
    >>> type_of_target(['a', 'b', 'c'])
    'multiclass'
    >>> type_of_target(np.array([[1, 2], [3, 1]]))
    'multiclass-multioutput'
    >>> type_of_target([[1, 2]]))
    'multiclass-multioutput'
    >>> type_of_target(np.array([[1.5, 2.0], [3.0, 1.6]]))
    'continuous-multioutput'
    >>> type_of_target(np.array([[0, 1], [1, 1]]))
    'multilabel-indicator'

    """
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
