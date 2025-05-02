import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import booleans, floats, integers
from scipy.sparse import csc_matrix

from pytorch_tabnet.utils.explain import explain_v1, explain_v2


@pytest.fixture
def explain_inputs():
    X = np.random.rand(10, 5).astype(np.float32)
    batch_size = 4
    device = torch.device("cpu")
    network = torch.nn.Module()  # Mock network
    normalize = False
    reducing_matrix = csc_matrix(np.random.rand(5, 3))  # Non-zero reducing matrix
    return X, batch_size, device, network, normalize, reducing_matrix


@pytest.mark.parametrize(
    "normalize,expected_sum",
    [
        (False, None),  # When normalize is False, we don't check sum
        (True, 1.0),  # When normalize is True, sum should be 1.0
    ],
)
def test_explain_v1_normalize(explain_inputs, normalize, expected_sum):
    X, batch_size, device, network, _, reducing_matrix = explain_inputs

    # Mock the network's forward_masks method with known values
    def mock_forward_masks(data):
        batch_size = data.shape[0]
        # Create non-zero positive values for explanations
        M_explain = torch.rand((batch_size, X.shape[1])) + 0.1
        masks = {"mask1": torch.rand((batch_size, X.shape[1])) + 0.1}
        return M_explain, masks

    network.forward_masks = mock_forward_masks

    res_explain, res_masks = explain_v1(X, batch_size, device, network, normalize, reducing_matrix)

    # Test v2 with numpy array
    res_explain_v2, res_masks_v2 = explain_v2(X, batch_size, device, network, normalize, reducing_matrix.toarray())

    # Compare v1 and v2 results
    np.testing.assert_allclose(res_explain, res_explain_v2, rtol=1e-5)
    for key in res_masks:
        np.testing.assert_allclose(res_masks[key], res_masks_v2[key], rtol=1e-5)

    assert isinstance(res_explain, np.ndarray)
    assert isinstance(res_masks, dict)
    assert not np.any(np.isnan(res_explain))  # Check for NaN values

    # Test shape of results
    assert res_explain.shape[0] == X.shape[0]  # Same number of samples
    assert res_explain.shape[1] == reducing_matrix.shape[1]  # Matches reduced dimensions

    # Test mask transformation
    expected_mask_shape = (X.shape[0], reducing_matrix.shape[1])  # Should match reduced dimensions
    assert res_masks["mask1"].shape == expected_mask_shape

    if expected_sum is not None:
        # Check if normalized rows sum to approximately 1
        np.testing.assert_allclose(np.sum(res_explain, axis=1), np.ones(res_explain.shape[0]), rtol=1e-5)


@given(
    X=arrays(
        dtype=np.float32,
        shape=integers(min_value=1, max_value=100).map(lambda n: (n, 5)),
        elements=floats(min_value=-1e3, max_value=1e3, allow_infinity=False, allow_nan=False),
    ),
    batch_size=integers(min_value=1, max_value=32),
    normalize=booleans(),
)
@settings(deadline=None, max_examples=1500)
def test_explain_v1_property_based(X, batch_size, normalize):
    device = torch.device("cpu")
    network = torch.nn.Module()
    reducing_matrix = csc_matrix(np.random.rand(X.shape[1], 3))  # Non-zero reducing matrix

    def mock_forward_masks(data):
        batch_size = data.shape[0]
        # Create non-zero positive values for explanations
        M_explain = torch.rand((batch_size, X.shape[1])) + 0.1
        masks = {"mask1": torch.rand((batch_size, X.shape[1])) + 0.1}
        return M_explain, masks

    network.forward_masks = mock_forward_masks

    res_explain, res_masks = explain_v1(X, batch_size, device, network, normalize, reducing_matrix.copy())

    # Test v2 with numpy array
    res_explain_v2, res_masks_v2 = explain_v2(X, batch_size, device, network, normalize, reducing_matrix.toarray())

    # Compare v1 and v2 results
    np.testing.assert_allclose(res_explain, res_explain_v2, rtol=1e-5)
    for key in res_masks:
        np.testing.assert_allclose(res_masks[key], res_masks_v2[key], rtol=1e-5)

    # Test properties that should always hold
    assert res_explain.shape[0] == X.shape[0]  # Same number of samples
    assert res_explain.shape[1] == reducing_matrix.shape[1]  # Matches reduced dimensions
    assert isinstance(res_masks, dict)
    assert "mask1" in res_masks
    assert res_masks["mask1"].shape == res_explain.shape
    assert not np.any(np.isnan(res_explain))  # No NaN values

    if normalize:
        # Check if normalized values are between 0 and 1
        assert np.all((res_explain >= 0) & (res_explain <= 1))
        # Check if rows sum to approximately 1
        np.testing.assert_allclose(np.sum(res_explain, axis=1), np.ones(res_explain.shape[0]), rtol=1e-5)
