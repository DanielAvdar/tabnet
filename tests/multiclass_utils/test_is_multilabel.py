import numpy as np
import scipy.sparse as sp

from pytorch_tabnet.utils.is_multilabel import is_multilabel


def test_is_multilabel_1d_array():
    """Test that 1D arrays are not multilabel."""
    y = np.array([0, 1, 0, 1])
    assert not is_multilabel(y)


def test_is_multilabel_list_of_lists():
    """Test that list of lists (ragged) is not multilabel."""
    y = [[1], [0, 2], []]
    assert not is_multilabel(y)


def test_is_multilabel_2d_binary():
    """Test that 2D binary arrays are multilabel."""
    y = np.array([[1, 0], [0, 1], [1, 1]])
    assert is_multilabel(y)


def test_is_multilabel_2d_single_column():
    """Test that 2D array with single column is not multilabel."""
    y = np.array([[1], [0], [0]])
    assert not is_multilabel(y)


def test_is_multilabel_2d_single_row():
    """Test that 2D array with single row but multiple columns is multilabel."""
    y = np.array([[1, 0, 0]])
    assert is_multilabel(y)


def test_is_multilabel_sparse_binary():
    """Test that sparse binary matrix is multilabel."""
    y = sp.csr_matrix([[1, 0], [0, 1], [1, 1]])
    assert is_multilabel(y)


def test_is_multilabel_non_binary():
    """Test that 2D non-binary arrays are not multilabel."""
    y = np.array([[1, 0], [0, 2], [3, 1]])
    assert not is_multilabel(y)


def test_is_multilabel_float_binary():
    """Test that 2D float arrays with only 0.0 and 1.0 are multilabel."""
    y = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert is_multilabel(y)
