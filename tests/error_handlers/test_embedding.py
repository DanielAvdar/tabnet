import pytest

from pytorch_tabnet.error_handlers.embedding_errors import check_embedding_parameters


class TestValidation:
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
