import numpy as np
import pytest
import scipy.sparse as sp

from pytorch_tabnet.utils.multiclass_validation import assert_all_finite, check_output_dim


def test_assert_all_finite_numpy():
    """Test assert_all_finite with numpy arrays."""
    # Normal case - should not raise
    X = np.array([1.0, 2.0, 3.0])
    assert_all_finite(X)

    # With NaN - should raise
    X = np.array([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="Input contains NaN, infinity"):
        assert_all_finite(X)

    # With infinity - should raise
    X = np.array([1.0, np.inf, 3.0])
    with pytest.raises(ValueError, match="Input contains NaN, infinity"):
        assert_all_finite(X)

    # With allow_nan=True but infinity - should raise
    X = np.array([1.0, np.inf, 3.0])
    with pytest.raises(ValueError, match="Input contains infinity"):
        assert_all_finite(X, allow_nan=True)


def test_assert_all_finite_sparse():
    """Test assert_all_finite with sparse matrices."""
    # Normal case - should not raise
    X = sp.csr_matrix([[1.0, 2.0], [3.0, 4.0]]).toarray()
    assert_all_finite(X)

    # With NaN - should raise
    X = sp.csr_matrix([[1.0, np.nan], [3.0, 4.0]]).toarray()
    with pytest.raises(ValueError, match="Input contains NaN, infinity"):
        assert_all_finite(X)

    # With infinity - should raise
    X = sp.csr_matrix([[1.0, np.inf], [3.0, 4.0]]).toarray()
    with pytest.raises(ValueError, match="Input contains NaN, infinity"):
        assert_all_finite(X)


def test_check_output_dim_valid():
    """Test check_output_dim with valid labels."""
    # All test labels present in training labels - should not raise
    train_labels = np.array([0, 1, 2, 3])
    test_labels = np.array([1, 2])
    check_output_dim(train_labels, test_labels)

    # None test array - should not raise
    test_labels = None
    check_output_dim(train_labels, test_labels)


def test_check_output_dim_invalid():
    """Test check_output_dim with invalid labels raises error."""
    # Test labels contain values not in training labels
    train_labels = np.array([0, 1, 2])
    test_labels = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="contains unkown targets from training"):
        check_output_dim(train_labels, test_labels)

    # Mixed types in test labels
    train_labels = np.array([0, 1, 2])
    test_labels = np.array([1, "a"], dtype=object)
    with pytest.raises(
        ValueError,
    ):
        check_output_dim(train_labels, test_labels)
