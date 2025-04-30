import numpy as np
import pytest

from pytorch_tabnet.multiclass_utils.dimension import infer_multitask_output, infer_output_dim


def test_infer_output_dim_binary():
    """Test infer_output_dim with binary classification data."""
    y_train = np.array([0, 1, 0, 1])
    output_dim, train_labels = infer_output_dim(y_train)
    assert output_dim == 2
    np.testing.assert_array_equal(train_labels, np.array([0, 1]))


def test_infer_output_dim_multiclass():
    """Test infer_output_dim with multiclass classification data."""
    y_train = np.array([0, 1, 2, 3, 1, 2, 0])
    output_dim, train_labels = infer_output_dim(y_train)
    assert output_dim == 4
    np.testing.assert_array_equal(train_labels, np.array([0, 1, 2, 3]))


def test_infer_output_dim_string_labels():
    """Test infer_output_dim with string labels."""
    y_train = np.array(["cat", "dog", "bird", "cat", "dog"])
    output_dim, train_labels = infer_output_dim(y_train)
    assert output_dim == 3
    np.testing.assert_array_equal(train_labels, np.array(["bird", "cat", "dog"]))


def test_infer_multitask_output_valid():
    """Test infer_multitask_output with valid multitask data."""
    y_train = np.array([[0, 1, 2], [1, 0, 1], [0, 1, 0], [1, 0, 2]])
    tasks_dims, tasks_labels = infer_multitask_output(y_train)
    assert tasks_dims == [2, 2, 3]
    np.testing.assert_array_equal(tasks_labels[0], np.array([0, 1]))
    np.testing.assert_array_equal(tasks_labels[1], np.array([0, 1]))
    np.testing.assert_array_equal(tasks_labels[2], np.array([0, 1, 2]))


def test_infer_multitask_output_invalid_shape():
    """Test infer_multitask_output with invalid shape (1D)."""
    y_train = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="y_train should be of shape"):
        infer_multitask_output(y_train)


def test_infer_multitask_output_task_failure():
    """Test infer_multitask_output with a task that would fail."""
    # We need to create task data that will fail but in a way the test can predict
    # Here we make a mixed int/float array for the second task which should fail when trying to infer output dim
    y_train = np.array([[0, 1.5], [1, 2.5], [0, 3.5], [1, 4.5]])
    with pytest.raises(ValueError, match="Error for task 1"):
        infer_multitask_output(y_train)
