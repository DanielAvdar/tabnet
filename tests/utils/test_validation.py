import numpy as np
import pytest

from pytorch_tabnet.utils.validation import (
    check_embedding_parameters,
    check_input,
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

        # Call without providing eval_name
        eval_name, validated_eval_set = validate_eval_set(eval_set, None, X_train, y_train)

        # Check results
        assert eval_name == ["val_0"]
        assert len(validated_eval_set) == 1
        assert np.array_equal(validated_eval_set[0][0], X_val)
        assert np.array_equal(validated_eval_set[0][1], y_val)

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
        validated_name, validated_eval_set = validate_eval_set(eval_set, eval_name, X_train, y_train)

        # Check results
        assert validated_name == ["custom_val"]
        assert len(validated_eval_set) == 1

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
        with pytest.raises(AssertionError, match="Each tuple of eval_set need to have two elements"):
            validate_eval_set(eval_set, None, X_train, y_train)

    def test_validate_eval_set_dimension_mismatch_X(self):
        """Test validate_eval_set with a dimension mismatch in X."""
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([0, 1, 0])

        # Create an invalid eval set with wrong X dimensions
        X_val = np.array([[7, 8, 9]])  # 3 columns instead of 2
        y_val = np.array([1])
        eval_set = [(X_val, y_val)]

        # Should raise AssertionError about dimension mismatch
        with pytest.raises(AssertionError, match="Number of columns is different between X_val_0"):
            validate_eval_set(eval_set, None, X_train, y_train)

    def test_validate_eval_set_dimension_mismatch_y_2d(self):
        """Test validate_eval_set with a dimension mismatch in y when y_train is 2D."""
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([[0, 1], [1, 0], [0, 0]])  # 2D y_train

        # Create an invalid eval set with wrong y dimensions
        X_val = np.array([[7, 8]])
        y_val = np.array([[1]])  # Only 1 column instead of 2
        eval_set = [(X_val, y_val)]

        # Should raise AssertionError about dimension mismatch
        with pytest.raises(AssertionError):
            validate_eval_set(eval_set, None, X_train, y_train)

    def test_validate_eval_set_row_count_mismatch(self):
        """Test validate_eval_set with a row count mismatch between X and y."""
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([0, 1, 0])

        # Create an invalid eval set with mismatched row counts
        X_val = np.array([[7, 8], [9, 10]])
        y_val = np.array([1])  # Only 1 row instead of 2
        eval_set = [(X_val, y_val)]

        # Should raise AssertionError about row count mismatch
        with pytest.raises(AssertionError, match="You need the same number of rows between X_val_0"):
            validate_eval_set(eval_set, None, X_train, y_train)

    def test_check_input_valid_numpy(self):
        """Test check_input with valid numpy array."""
        X = np.array([[1, 2], [3, 4]])
        # Should not raise any exceptions
        check_input(X)

    def test_check_embedding_parameters_valid(self):
        """Test check_embedding_parameters with valid inputs."""
        cat_dims = [3, 4, 5]  # Cardinality of each categorical feature
        cat_idxs = [0, 1, 2]  # Indices of categorical features
        cat_emb_dim = 2  # Embedding dimension (same for all)

        result_dims, result_idxs, result_emb_dims = check_embedding_parameters(cat_dims, cat_idxs, cat_emb_dim)

        # Check results
        assert result_dims == cat_dims
        assert result_idxs == cat_idxs
        assert result_emb_dims == [2, 2, 2]

    def test_check_embedding_parameters_list_emb_dim(self):
        """Test check_embedding_parameters with list for embedding dimensions."""
        cat_dims = [3, 4, 5]
        cat_idxs = [0, 1, 2]
        cat_emb_dim = [2, 3, 4]  # Different embedding dimension for each

        result_dims, result_idxs, result_emb_dims = check_embedding_parameters(cat_dims, cat_idxs, cat_emb_dim)

        # Check results
        assert result_dims == cat_dims
        assert result_idxs == cat_idxs
        assert result_emb_dims == cat_emb_dim

    def test_check_embedding_parameters_reordering(self):
        """Test check_embedding_parameters with unsorted indices."""
        cat_dims = [3, 4, 5]
        cat_idxs = [2, 0, 1]  # Unsorted indices
        cat_emb_dim = [2, 3, 4]

        result_dims, result_idxs, result_emb_dims = check_embedding_parameters(cat_dims, cat_idxs, cat_emb_dim)

        # Check results - should be sorted by cat_idxs
        expected_order = [1, 2, 0]  # New positions after sorting by cat_idxs
        expected_dims = [cat_dims[i] for i in expected_order]
        expected_emb_dims = [cat_emb_dim[i] for i in expected_order]

        assert result_dims == expected_dims
        assert result_idxs == cat_idxs  # Indices don't change
        assert result_emb_dims == expected_emb_dims

    def test_check_embedding_parameters_empty_lists(self):
        """Test check_embedding_parameters with empty lists."""
        cat_dims = []
        cat_idxs = []
        cat_emb_dim = []

        result_dims, result_idxs, result_emb_dims = check_embedding_parameters(cat_dims, cat_idxs, cat_emb_dim)

        # Check results
        assert result_dims == []
        assert result_idxs == []
        assert result_emb_dims == []

    def test_check_embedding_parameters_missing_dims(self):
        """Test check_embedding_parameters with missing cat_dims."""
        cat_dims = []
        cat_idxs = [0, 1]  # Non-empty
        cat_emb_dim = 2

        with pytest.raises(ValueError, match="If cat_idxs is non-empty, cat_dims must be defined"):
            check_embedding_parameters(cat_dims, cat_idxs, cat_emb_dim)

    def test_check_embedding_parameters_missing_idxs(self):
        """Test check_embedding_parameters with missing cat_idxs."""
        cat_dims = [3, 4]  # Non-empty
        cat_idxs = []
        cat_emb_dim = 2

        with pytest.raises(ValueError, match="If cat_dims is non-empty, cat_idxs must be defined"):
            check_embedding_parameters(cat_dims, cat_idxs, cat_emb_dim)

    def test_check_embedding_parameters_length_mismatch_dims_idxs(self):
        """Test check_embedding_parameters with mismatched length of cat_dims and cat_idxs."""
        cat_dims = [3, 4, 5]
        cat_idxs = [0, 1]  # Different length
        cat_emb_dim = 2

        with pytest.raises(ValueError, match="The lists cat_dims and cat_idxs must have the same length"):
            check_embedding_parameters(cat_dims, cat_idxs, cat_emb_dim)

    def test_check_embedding_parameters_length_mismatch_emb_dims(self):
        """Test check_embedding_parameters with mismatched length of cat_emb_dim and cat_dims."""
        cat_dims = [3, 4, 5]
        cat_idxs = [0, 1, 2]
        cat_emb_dim = [2, 3]  # Different length

        with pytest.raises(ValueError, match="cat_emb_dim and cat_dims must be lists of same length"):
            check_embedding_parameters(cat_dims, cat_idxs, cat_emb_dim)
