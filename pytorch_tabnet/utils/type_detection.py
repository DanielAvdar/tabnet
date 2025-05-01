"""Type detection utilities for TabNet."""

from typing import List

import numpy as np

from pytorch_tabnet.utils.type_of_target import type_of_target


def _get_allowed_classification_types() -> List[str]:
    return [
        "binary",
        "multiclass",
        "multiclass-multioutput",
        "multilabel-indicator",
        "multilabel-sequences",
    ]


def _is_valid_classification_type(y_type: str) -> bool:
    return y_type in _get_allowed_classification_types()


def check_classification_targets(y: np.ndarray) -> None:
    """Ensure that target y is of a non-regression type.

    Only the following target types (as defined in type_of_target) are allowed:
        'binary', 'multiclass', 'multiclass-multioutput',
        'multilabel-indicator', 'multilabel-sequences'

    Parameters
    ----------
    y : array-like

    """
    y_type = type_of_target(y)
    if not _is_valid_classification_type(y_type):
        raise ValueError("Unknown label type: %r" % y_type)
