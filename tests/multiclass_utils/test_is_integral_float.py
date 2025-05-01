import numpy as np

from pytorch_tabnet.utils._is_integral_float import _is_integral_float


def test_is_integral_float_true():
    """Test that float arrays with integral values return True."""
    y = np.array([1.0, 2.0, 3.0])
    assert _is_integral_float(y)


def test_is_integral_float_false():
    """Test that float arrays with non-integral values return False."""
    y = np.array([1.0, 2.5, 3.0])
    assert not _is_integral_float(y)


def test_is_integral_float_int_array():
    """Test that integer arrays return False (not float dtype)."""
    y = np.array([1, 2, 3])
    assert not _is_integral_float(y)
