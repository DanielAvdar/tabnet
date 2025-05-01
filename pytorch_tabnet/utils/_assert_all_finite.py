import numpy as np


def _is_float_dtype(X: np.ndarray) -> bool:
    return X.dtype.kind in "fc"


def _has_finite_sum(X: np.ndarray) -> bool:
    return np.isfinite(np.sum(X))


def _check_float_array(X: np.ndarray, allow_nan: bool) -> None:
    msg_err = "Input contains {} or a value too large for {!r}."
    has_inf = np.isinf(X).any()
    has_nan = not np.isfinite(X).all()

    if (allow_nan and has_inf) or (not allow_nan and has_nan):
        type_err = "infinity" if allow_nan else "NaN, infinity"
        raise ValueError(msg_err.format(type_err, X.dtype))


def _check_object_array_for_nan(X: np.ndarray, allow_nan: bool) -> None:
    if not allow_nan and np.isnan(X).any():
        raise ValueError("Input contains NaN")


def _assert_all_finite(X: np.ndarray, allow_nan: bool = False) -> None:
    """Like assert_all_finite, but only for ndarray."""
    X = np.asanyarray(X)

    is_float = _is_float_dtype(X)

    if is_float and _has_finite_sum(X):
        pass
    elif is_float:
        _check_float_array(X, allow_nan)
    # for object dtype data, we only check for NaNs (GH-13254)
    elif X.dtype == np.dtype("object") and not allow_nan:
        _check_object_array_for_nan(X, allow_nan)
