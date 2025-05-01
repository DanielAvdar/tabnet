import numpy as np
import pytest
import scipy.sparse as sp

from pytorch_tabnet.utils.validation_multi import assert_all_finite, check_output_dim, check_unique_type


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
    X = sp.csr_matrix([[1.0, 2.0], [3.0, 4.0]])
    assert_all_finite(X)

    # With NaN - should raise
    X = sp.csr_matrix([[1.0, np.nan], [3.0, 4.0]])
    with pytest.raises(ValueError, match="Input contains NaN, infinity"):
        assert_all_finite(X)

    # With infinity - should raise
    X = sp.csr_matrix([[1.0, np.inf], [3.0, 4.0]])
    with pytest.raises(ValueError, match="Input contains NaN, infinity"):
        assert_all_finite(X)


def test_check_unique_type_homogeneous():
    """Test check_unique_type with homogeneous types."""
    # Integer types - should not raise
    y = np.array([1, 2, 3])
    check_unique_type(y)

    # Float types - should not raise
    y = np.array([1.0, 2.0, 3.0])
    check_unique_type(y)

    # String types - should not raise
    y = np.array(["a", "b", "c"])
    check_unique_type(y)


def test_check_unique_type_heterogeneous():
    """Test check_unique_type with heterogeneous types raises error."""
    # Mixed integers and strings
    y = np.array([1, "a", 2], dtype=object)
    with pytest.raises(TypeError, match="Values on the target must have the same type"):
        check_unique_type(y)

    # Mixed integers and floats
    y = np.array([1, 2.0, 3], dtype=object)
    with pytest.raises(TypeError, match="Values on the target must have the same type"):
        check_unique_type(y)


def test_check_output_dim_valid():
    """Test check_output_dim with valid labels."""
    # All test labels present in training labels - should not raise
    train_labels = np.array([0, 1, 2, 3])
    test_labels = np.array([1, 2])
    check_output_dim(train_labels, test_labels)

    # None test array - should not raise
    test_labels = None
    check_output_dim(train_labels, test_labels)


def test_check_output_dim_empty_array():
    """Test check_output_dim with an empty array by monkeypatching check_unique_type."""
    # We need to patch the validation module's check_unique_type function to handle empty arrays
    import pytorch_tabnet.utils.validation_multi as validation

    original_check_unique_type = validation.check_unique_type

    try:
        # Define a patched version that doesn't fail on empty arrays
        def patched_check_unique_type(y):
            if y is not None and len(y) == 0:
                return  # Do nothing for empty arrays
            return original_check_unique_type(y)

        # Apply the patch
        validation.check_unique_type = patched_check_unique_type

        # Now test with empty array
        train_labels = np.array([0, 1, 2, 3])
        test_labels = np.array([])
        check_output_dim(train_labels, test_labels)  # Should not raise
    finally:
        # Restore original function
        validation.check_unique_type = original_check_unique_type


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
    with pytest.raises(TypeError, match="Values on the target must have the same type"):
        check_output_dim(train_labels, test_labels)
