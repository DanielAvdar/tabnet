import pytest
import numpy as np
from sklearn.utils.multiclass import (
    unique_labels,
    type_of_target,
    # infer_output_dim,
    # check_output_dim,
    # infer_multitask_output,
)
from scipy import sparse

from pytorch_tabnet.multiclass_utils import infer_multitask_output, check_output_dim, infer_output_dim
from pytorch_tabnet.multiclass_utils import assert_all_finite


# Tests for unique_labels
def test_unique_labels_empty():
    with pytest.raises(ValueError):
        unique_labels()


def test_unique_labels_mixed_types():
    with pytest.raises(ValueError):
        unique_labels([1, 2, "a"])


def test_unique_labels_inferred():
    assert np.array_equal(unique_labels(np.array([1, 2, 3])), np.array([1, 2, 3]))


# Tests for type_of_target
@pytest.mark.parametrize(
    "y,expected",
    [
        ([0.1, 0.6], "continuous"),
        ([1, -1, -1, 1], "binary"),
        (["a", "b", "a"], "binary"),
        ([1.0, 2.0], "binary"),
        ([1, 0, 2], "multiclass"),
        ([1.0, 0.0, 3.0], "multiclass"),
        (["a", "b", "c"], "multiclass"),
        (np.array([[1, 2], [3, 1]]), "multiclass-multioutput"),
        ([[1, 2]], "multilabel-indicator"),
        (np.array([[1.5, 2.0], [3.0, 1.6]]), "continuous-multioutput"),
        (np.array([[0, 1], [1, 1]]), "multilabel-indicator"),
    ],
)
def test_type_of_target_valid(y, expected):
    assert type_of_target(y) == expected





# Tests for infer_output_dim
def test_infer_output_dim_single_task():

    y = np.array([1, 0, 3, 3, 0, 1])
    output_dim, labels = infer_output_dim(y)
    assert output_dim == 3
    assert np.array_equal(labels, [0, 1, 3])


# Tests for check_output_dim
def test_check_output_dim_valid():
    labels = [0, 1, 2]
    y = np.array([0, 1, 1, 2, 0])

    check_output_dim(labels, y)  # No assertion, should pass


def test_check_output_dim_invalid():
    labels = [0, 1, 2]
    y = np.array([0, 1, 4])
    with pytest.raises(ValueError):
        check_output_dim(labels, y)



# Tests for infer_multitask_output
def test_infer_multitask_output_valid():
    y = np.array([[0, 1], [1, 2], [2, 0], [1, 1]])

    task_dims, task_labels = infer_multitask_output(y)
    assert task_dims == [3, 3]
    assert np.array_equal(task_labels[0], [0, 1, 2])
    assert np.array_equal(task_labels[1], [0, 1, 2])


def test_infer_multitask_output_invalid_shape():

    y = np.array([0, 1, 2])
    with pytest.raises(ValueError):
        infer_multitask_output(y)





@pytest.mark.parametrize(
    "X, allow_nan, raises_error",
    [
        # Dense array with no NaN or Inf
        (np.array([[1, 2], [3, 4]]), False, False),
        # Dense array with NaN
        (np.array([[1, np.nan], [3, 4]]), False, True),
        # Dense array with Inf
        (np.array([[1, np.inf], [3, 4]]), False, True),
        # Dense array with NaN, but allow_nan=True
        (np.array([[1, np.nan], [3, 4]]), True, False),
        # Sparse matrix with no NaN or Inf
        (sparse.csr_matrix([[1, 0], [0, 4]]), False, False),
        # Sparse matrix with NaN
        (sparse.csr_matrix([[1, np.nan], [0, 4]]), False, True),
        # Sparse matrix with Inf
        (sparse.csr_matrix([[1, np.inf], [0, 4]]), False, True),
        # Sparse matrix with NaN, but allow_nan=True
        (sparse.csr_matrix([[1, np.nan], [0, 4]]), True, False),
    ],
)
def test_assert_all_finite(X, allow_nan, raises_error):
    if raises_error:
        with pytest.raises(ValueError):
            assert_all_finite(X, allow_nan=allow_nan)
    else:
        assert_all_finite(X, allow_nan=allow_nan)