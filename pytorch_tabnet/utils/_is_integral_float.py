import numpy as np


def _is_integral_float(y: np.ndarray) -> bool:
    return y.dtype.kind == "f" and np.all(y.astype(int) == y)
