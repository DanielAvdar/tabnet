import pytest
import torch

# Determine available devices for parameterization
_devices = ["cpu"]
if torch.cuda.is_available():
    _devices.append("cuda")
try:
    if torch.backends.mps.is_available():
        _devices.append("mps")
except Exception:
    pass


@pytest.fixture(params=_devices)
def device_str(request):
    """Fixture that yields 'cpu', 'cuda' (if available), and 'mps' (if available and torch>=2.7.0)."""
    return request.param


def test_device_str(device_str):
    """Test to collect device string."""
