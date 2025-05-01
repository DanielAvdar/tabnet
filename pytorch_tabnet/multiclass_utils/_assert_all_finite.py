"""Import from utils module for backward compatibility."""

from pytorch_tabnet.utils._assert_all_finite import _assert_all_finite as _assert_all_finite

# For backward compatibility with tests that access this as a module attribute
_assert_all_finite._assert_all_finite = _assert_all_finite
