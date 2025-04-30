import numpy as np


def _assert_all_finite(X: np.ndarray, allow_nan: bool = False) -> None:
    """Like assert_all_finite, but only for ndarray."""
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method. The sum is also calculated
    # safely to reduce dtype induced overflows.
    is_float = X.dtype.kind in "fc"
    if is_float and (np.isfinite(np.sum(X))):
        pass
    elif is_float:
        msg_err = "Input contains {} or a value too large for {!r}."
        if allow_nan and np.isinf(X).any() or not allow_nan and not np.isfinite(X).all():
            type_err = "infinity" if allow_nan else "NaN, infinity"
            raise ValueError(msg_err.format(type_err, X.dtype))
    # for object dtype data, we only check for NaNs (GH-13254)
    elif X.dtype == np.dtype("object") and not allow_nan:
        if np.isnan(X).any():
            raise ValueError("Input contains NaN")
