import pytest
import torch

from pytorch_tabnet.error_handlers.validation import check_list_groups
from pytorch_tabnet.utils.matrices import _create_explain_matrix as create_explain_matrix  # noqa:
from pytorch_tabnet.utils.matrices import create_group_matrix


class TestMatrices:
    """Tests for the matrices utility functions."""

    def test_create_explain_matrix_uniform_emb_dim(self):
        """Test create_explain_matrix with uniform embedding dimension."""
        input_dim = 5
        cat_emb_dim = 3  # Same embedding dimension for all categorical features
        cat_idxs = [1, 3]  # Features at indices 1 and 3 are categorical
        post_embed_dim = 9  # 3 features remain scalar, 2 features expand to dim 3 each: 3 + 2*3 = 9

        result = create_explain_matrix(input_dim, cat_emb_dim, cat_idxs, post_embed_dim)

        # Verify type

        # Verify shape
        assert result.shape == (post_embed_dim, input_dim)

    def test_create_group_matrix_empty_groups(self):
        """Test create_group_matrix with no groups specified."""
        input_dim = 5
        list_groups = []  # No groups

        result = create_group_matrix(list_groups, input_dim)

        # Should return identity matrix
        expected = torch.eye(input_dim)
        assert torch.equal(result, expected)

    def test_create_group_matrix_with_groups(self):
        """Test create_group_matrix with specified groups."""
        input_dim = 6
        list_groups = [[0, 1, 2], [4, 5]]  # Features 0,1,2 in one group, 4,5 in another, 3 alone

        result = create_group_matrix(list_groups, input_dim)

        # Expected shape: (3 groups, 6 features)
        assert result.shape == (3, 6)

        # First group: features 0,1,2 have equal importance (1/3)
        assert result[0, 0] == pytest.approx(1 / 3)
        assert result[0, 1] == pytest.approx(1 / 3)
        assert result[0, 2] == pytest.approx(1 / 3)
        assert result[0, 3] == 0
        assert result[0, 4] == 0
        assert result[0, 5] == 0

        # Second group: features 4,5 have equal importance (1/2)
        assert result[1, 0] == 0
        assert result[1, 1] == 0
        assert result[1, 2] == 0
        assert result[1, 3] == 0
        assert result[1, 4] == pytest.approx(1 / 2)
        assert result[1, 5] == pytest.approx(1 / 2)

        # Third group: feature 3 alone, importance 1
        assert result[2, 0] == 0
        assert result[2, 1] == 0
        assert result[2, 2] == 0
        assert result[2, 3] == 1
        assert result[2, 4] == 0
        assert result[2, 5] == 0

        # Each row (group) should sum to 1
        for i in range(result.shape[0]):
            assert torch.sum(result[i, :]).item() == pytest.approx(1.0)

    def test_check_list_groups_valid(self):
        """Test check_list_groups with valid input."""
        input_dim = 6
        list_groups = [[0, 1, 2], [4, 5]]

        # Should not raise any exceptions
        check_list_groups(list_groups, input_dim)

    def test_check_list_groups_empty(self):
        """Test check_list_groups with empty list."""
        input_dim = 6
        list_groups = []

        # Should not raise any exceptions
        check_list_groups(list_groups, input_dim)

    def test_check_list_groups_not_list(self):
        """Test check_list_groups with non-list input."""
        input_dim = 6
        list_groups = "not a list"

        with pytest.raises(AssertionError, match="list_groups must be a list of list."):
            check_list_groups(list_groups, input_dim)

    def test_check_list_groups_group_not_list(self):
        """Test check_list_groups with a group that is not a list."""
        input_dim = 6
        list_groups = [[0, 1], "not a list"]

        with pytest.raises(AssertionError, match="Groups must be given as a list of list"):
            check_list_groups(list_groups, input_dim)

    def test_check_list_groups_empty_group(self):
        """Test check_list_groups with an empty group."""
        input_dim = 6
        list_groups = [[0, 1], []]

        with pytest.raises(AssertionError, match="Empty groups are forbidding"):
            check_list_groups(list_groups, input_dim)

    def test_check_list_groups_duplicate_feature(self):
        """Test check_list_groups with a feature in multiple groups."""
        input_dim = 6
        list_groups = [[0, 1], [1, 2]]  # Feature 1 appears in both groups

        with pytest.raises(AssertionError, match="One feature can only appear in one group"):
            check_list_groups(list_groups, input_dim)

    def test_check_list_groups_invalid_feature(self):
        """Test check_list_groups with feature index >= input_dim."""
        input_dim = 6
        list_groups = [[0, 1], [6, 7]]  # Features 6,7 exceed input_dim

        with pytest.raises(AssertionError, match="Number of features is"):
            check_list_groups(list_groups, input_dim)
