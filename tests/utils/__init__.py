from pytorch_tabnet.utils import (
    ComplexEncoder,
    check_embedding_parameters,
    check_input,
    create_explain_matrix,
    create_group_matrix,
    define_device,
    filter_weights,
    validate_eval_set,
)


class TestInit:
    """Tests for the utils module initialization."""

    def test_imports(self):
        """Test that all utility functions are correctly imported."""
        # Verify that all imports are working and pointing to the correct functions/classes
        assert callable(define_device)
        assert callable(create_explain_matrix)
        assert callable(create_group_matrix)
        assert isinstance(ComplexEncoder, type)
        assert callable(check_embedding_parameters)
        assert callable(check_input)
        assert callable(filter_weights)
        assert callable(validate_eval_set)
