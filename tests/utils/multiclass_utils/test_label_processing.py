import numpy as np
import pytest

from pytorch_tabnet.utils.label_processing import _unique_multiclass, unique_labels


def test_unique_labels_single_array():
    """Test unique_labels with a single array."""
    y = np.array([3, 5, 5, 5, 7, 7])
    result = unique_labels(y)
    np.testing.assert_array_equal(result, np.array([3, 5, 7]))


def test_unique_labels_multiple_arrays():
    """Test unique_labels with multiple arrays."""
    y1 = np.array([1, 2, 3, 4])
    y2 = np.array([2, 2, 3, 4])
    result = unique_labels(y1, y2)
    np.testing.assert_array_equal(result, np.array([1, 2, 3, 4]))


def test_unique_labels_disjoint_arrays():
    """Test unique_labels with disjoint arrays."""
    y1 = np.array([1, 2, 10])
    y2 = np.array([5, 11])
    result = unique_labels(y1, y2)
    np.testing.assert_array_equal(result, np.array([1, 2, 5, 10, 11]))


def test_unique_labels_string_arrays():
    """Test unique_labels with string arrays."""
    y = np.array(["cat", "dog", "cat", "bird"])
    result = unique_labels(y)
    np.testing.assert_array_equal(result, np.array(["bird", "cat", "dog"]))


def test_unique_labels_mixed_types():
    """Test unique_labels with mixed string and number types raises error."""
    y1 = np.array(["cat", "dog"])
    y2 = np.array([1, 2])
    with pytest.raises(ValueError, match="Mix of label input types"):
        unique_labels(y1, y2)


def test_unique_labels_no_args():
    """Test unique_labels with no arguments raises error."""
    with pytest.raises(ValueError, match="No argument has been passed"):
        unique_labels()


def test_unique_labels_mixed_target_types():
    """Test unique_labels with mixed target types raises error."""
    y1 = np.array([1, 2, 3])  # multiclass
    y2 = np.array([[1, 0], [0, 1], [1, 1]])  # multilabel-indicator
    with pytest.raises(ValueError, match="Mix type of y not allowed"):
        unique_labels(y1, y2)


def test_unique_labels_multilabel_indicator():
    """Test unique_labels with multilabel indicator raises error."""
    y = np.array([[1, 0], [0, 1], [1, 1]])
    with pytest.raises(IndexError, match="If attempting multilabel classification"):
        unique_labels(y)


def test_unique_multiclass_non_array_like():
    """Test _unique_multiclass with object that doesn't have __array__ attribute."""

    # Using a simple set as input instead of a numpy array
    y = {1, 2, 3}  # set doesn't have __array__ attribute
    result = _unique_multiclass(y)
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))
