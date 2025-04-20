import pytest
import torch
from activations_plus import Entmax as Entmax15
from activations_plus import Sparsemax

# from pytorch_tabnet.sparsemax import entmoid15


# Generate test cases for Sparsemax
@pytest.mark.parametrize("dim", [-1, 0, 1])
@pytest.mark.parametrize("input_shape", [(5, 3), (2, 4, 6), (2, 2, 3, 4), (10, 20, 30)])
def test_sparsemax(input_shape, dim):
    input_ = torch.randn(input_shape)
    sparsemax_op = Sparsemax(dim=dim)
    output = sparsemax_op(input_)

    # Check shape
    assert output.shape == input_.shape

    # Check normalization along the specified dimension
    sum_along_dim = output.sum(dim=dim)
    assert torch.allclose(sum_along_dim, torch.ones_like(sum_along_dim))

    # Check sparsity (some elements should be exactly zero)
    assert (output == 0).any()


# Generate test cases for Entmax15
@pytest.mark.parametrize("dim", [-1, 0, 1])
@pytest.mark.parametrize("input_shape", [(5, 3), (2, 4, 6)])
def test_entmax15(input_shape, dim):
    input_ = torch.randn(input_shape)
    entmax15_op = Entmax15(dim=dim)
    output = entmax15_op(input_)

    # Check shape
    assert output.shape == input_.shape

    # Check normalization along the specified dimension
    sum_along_dim = output.sum(dim=dim)
    assert torch.allclose(sum_along_dim, torch.ones_like(sum_along_dim))


