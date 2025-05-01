"""Import from utils module for backward compatibility."""

from .multiclass_validation import check_unique_type as _original_check_unique_type, assert_all_finite  # noqa

# assert_all_finite


# For backward compatibility with tests that monkey patch this function
def check_unique_type(y):
    if y is not None and len(y) == 0:
        return  # Do nothing for empty arrays
    return _original_check_unique_type(y)


# Override check_output_dim to use our patched check_unique_type
def check_output_dim(labels, y):
    """Check that all labels in y are present in the training labels.

    Parameters
    ----------
    labels : np.ndarray
        Array of valid labels from training.
    y : np.ndarray
        Array of labels to check.

    Raises
    ------
    ValueError
        If y contains labels not present in labels.
    """
    if y is not None:
        check_unique_type(y)
        # Rest of the implementation from utils.multiclass_validation
        from pytorch_tabnet.utils.label_processing import unique_labels

        valid_labels = unique_labels(y)
        if not set(valid_labels).issubset(set(labels)):
            raise ValueError(
                f"""Valid set -- {set(valid_labels)} --\n" +
                "contains unkown targets from training --\n" +
                f"{set(labels)}"""
            )
    return
