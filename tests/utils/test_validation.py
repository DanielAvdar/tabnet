import numpy as np
import pytest
from sklearn.utils import check_array

from pytorch_tabnet.utils.validation import (
    filter_weights,
    validate_eval_set,
)


class TestValidation:
    """Tests for the validation utility functions."""

    def test_filter_weights_valid_list(self):
        """Test filter_weights with a valid list."""
        # This should not raise any exceptions
        filter_weights([1, 2, 3])

    def test_filter_weights_valid_array(self):
        """Test filter_weights with a valid numpy array."""
        # This should not raise any exceptions
        filter_weights(np.array([1, 2, 3]))

    def test_filter_weights_invalid_int(self):
        """Test filter_weights with an invalid integer value."""
        with pytest.raises(
            ValueError, match="Please provide a list or np.array of weights for regression, multitask or pretraining: 1 given."
        ):
            filter_weights(1)

    def test_filter_weights_invalid_dict(self):
        """Test filter_weights with an invalid dictionary value."""
        with pytest.raises(
            ValueError, match="Please provide a list or np.array of weights for regression, multitask or pretraining: Dict given."
        ):
            filter_weights({"a": 1, "b": 2})

    def test_validate_eval_set_valid(self):
        """Test validate_eval_set with valid inputs."""
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([0, 1, 0])

        # Create a valid eval set
        X_val = np.array([[7, 8], [9, 10]])
        y_val = np.array([1, 0])
        eval_set = [(X_val, y_val)]
        eval_name = [f"val_{i}" for i in range(len(eval_set))]

        # Call without providing eval_name
        validate_eval_set(eval_set, eval_name, X_train, y_train)

        # Check results
        assert eval_name == ["val_0"]
        assert len(eval_set) == 1
        assert np.array_equal(eval_set[0][0], X_val)
        assert np.array_equal(eval_set[0][1], y_val)

    def test_validate_eval_set_with_custom_names(self):
        """Test validate_eval_set with custom evaluation set names."""
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([0, 1, 0])

        # Create a valid eval set
        X_val = np.array([[7, 8], [9, 10]])
        y_val = np.array([1, 0])
        eval_set = [(X_val, y_val)]
        eval_name = ["custom_val"]

        # Call with custom eval_name
        validate_eval_set(eval_set, eval_name, X_train, y_train)

        # Check results
        assert eval_name == ["custom_val"]
        assert len(eval_set) == 1

    def test_validate_eval_set_mismatched_name_length(self):
        """Test validate_eval_set with mismatched lengths of eval_set and eval_name."""
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([0, 1, 0])

        # Create a valid eval set
        X_val = np.array([[7, 8], [9, 10]])
        y_val = np.array([1, 0])
        eval_set = [(X_val, y_val)]
        eval_name = ["name1", "name2"]  # Too many names

        # Should raise AssertionError about lengths
        with pytest.raises(AssertionError, match="eval_set and eval_name have not the same length"):
            validate_eval_set(eval_set, eval_name, X_train, y_train)

    def test_validate_eval_set_invalid_tuple_length(self):
        """Test validate_eval_set with tuples that don't have exactly two elements."""
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([0, 1, 0])

        # Create an invalid eval set with tuples that have wrong number of elements
        eval_set = [(np.array([[7, 8]]), np.array([1]), np.array([2]))]  # Tuple with 3 elements

        # Should raise AssertionError about tuple lengths
        eval_names = [f"val_{i}" for i in range(len(eval_set))]
        with pytest.raises(AssertionError, match="Each tuple of eval_set need to have two elements"):
            validate_eval_set(eval_set, eval_names, X_train, y_train)

    def test_validate_eval_set_dimension_mismatch_X(self):
        """Test validate_eval_set with a dimension mismatch in X."""
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([0, 1, 0])

        # Create an invalid eval set with wrong X dimensions
        X_val = np.array([[7, 8, 9]])  # 3 columns instead of 2
        y_val = np.array([1])
        eval_set = [(X_val, y_val)]
        eval_names = [f"val_{i}" for i in range(len(eval_set))]

        # Should raise AssertionError about dimension mismatch
        with pytest.raises(AssertionError, match="Number of columns is different between X_val_0"):
            validate_eval_set(eval_set, eval_names, X_train, y_train)

    def test_validate_eval_set_dimension_mismatch_y_2d(self):
        """Test validate_eval_set with a dimension mismatch in y when y_train is 2D."""
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([[0, 1], [1, 0], [0, 0]])  # 2D y_train

        # Create an invalid eval set with wrong y dimensions
        X_val = np.array([[7, 8]])
        y_val = np.array([[1]])  # Only 1 column instead of 2
        eval_set = [(X_val, y_val)]
        eval_names = [f"val_{i}" for i in range(len(eval_set))]

        # Should raise AssertionError about dimension mismatch
        with pytest.raises(AssertionError):
            validate_eval_set(eval_set, eval_names, X_train, y_train)

    def test_validate_eval_set_row_count_mismatch(self):
        """Test validate_eval_set with a row count mismatch between X and y."""
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([0, 1, 0])

        # Create an invalid eval set with mismatched row counts
        X_val = np.array([[7, 8], [9, 10]])
        y_val = np.array([1])  # Only 1 row instead of 2
        eval_set = [(X_val, y_val)]
        eval_names = [f"val_{i}" for i in range(len(eval_set))]

        # Should raise AssertionError about row count mismatch
        with pytest.raises(AssertionError, match="You need the same number of rows between X_val_0"):
            validate_eval_set(eval_set, eval_names, X_train, y_train)

    def test_check_input_valid_numpy(self):
        """Test check_input with valid numpy array."""
        X = np.array([[1, 2], [3, 4]])
        # Should not raise any exceptions
        check_array(X)
