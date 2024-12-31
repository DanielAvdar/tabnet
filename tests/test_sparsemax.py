import torch
import pytest

from pytorch_tabnet.sparsemax import Sparsemax, entmoid15


# Generate test cases for Sparsemax
@pytest.mark.parametrize("dim", [-1, 0, 1])
@pytest.mark.parametrize("input_shape", [(5, 3), (2, 4, 6)])
def test_sparsemax(input_shape, dim):
    input = torch.randn(input_shape)
    sparsemax_op = Sparsemax(dim=dim)
    output = sparsemax_op(input)

    # Check shape
    assert output.shape == input.shape

    # Check normalization along the specified dimension
    sum_along_dim = output.sum(dim=dim)
    assert torch.allclose(sum_along_dim, torch.ones_like(sum_along_dim))

    # Check sparsity (some elements should be exactly zero)
    assert (output == 0).any()

# Generate test cases for Entmax15
@pytest.mark.parametrize("dim", [-1, 0, 1])
@pytest.mark.parametrize("input_shape", [(5, 3), (2, 4, 6)])
def test_entmax15(input_shape, dim):
    input = torch.randn(input_shape)
    entmax15_op = Entmax15(dim=dim)
    output = entmax15_op(input)

    # Check shape
    assert output.shape == input.shape

    # Check normalization along the specified dimension
    sum_along_dim = output.sum(dim=dim)
    assert torch.allclose(sum_along_dim, torch.ones_like(sum_along_dim))


# Generate test cases for entmoid15
@pytest.mark.parametrize("input_tensor", [torch.randn(5, 3), torch.randn(2, 4, 6)])
def test_entmoid15(input_tensor):
    output = entmoid15(input_tensor)
    # Check value range
    assert ((output >= 0) & (output <= 1)).all()