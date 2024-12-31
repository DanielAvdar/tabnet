import pytest
import numpy as np
from sklearn.utils.multiclass import (
    unique_labels,
    type_of_target,
    # infer_output_dim,
    # check_output_dim,
    # infer_multitask_output,
)

from pytorch_tabnet.multiclass_utils import infer_multitask_output, check_output_dim, infer_output_dim


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