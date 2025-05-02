"""Dimension inference utilities for TabNet."""

from typing import List, Tuple

import numpy as np

from pytorch_tabnet.utils.label_processing import unique_labels


def infer_output_dim(y_train: np.ndarray) -> tuple[int, np.ndarray]:
    """Infer output_dim from targets.

    Parameters
    ----------
    y_train : np.array
        Training targets

    Returns
    -------
    output_dim : int
        Number of classes for output
    train_labels : list
        Sorted list of initial classes

    """
    train_labels = unique_labels(y_train)
    output_dim = len(train_labels)

    return output_dim, train_labels


def _validate_multitask_shape(y_train: np.ndarray) -> None:
    if len(y_train.shape) < 2:
        raise ValueError("y_train should be of shape (n_examples, n_tasks)" + f"but got {y_train.shape}")


def _infer_single_task(y_task: np.ndarray, task_idx: int) -> Tuple[int, np.ndarray]:
    try:
        output_dim, train_labels = infer_output_dim(y_task)
        return output_dim, train_labels
    except ValueError as err:
        raise ValueError(f"""Error for task {task_idx} : {err}""")


def infer_multitask_output(y_train: np.ndarray) -> tuple[List[int], List[np.ndarray]]:
    """Infer output_dim and label sets for multitask targets.

    This is for multiple tasks.

    Parameters
    ----------
    y_train : np.ndarray
        Training targets, shape (n_examples, n_tasks)

    Returns
    -------
    tasks_dims : list of int
        Number of classes for each output
    tasks_labels : list of np.ndarray
        List of sorted list of initial classes for each task

    Raises
    ------
    ValueError
        If y_train does not have at least 2 dimensions or a task fails.

    """
    _validate_multitask_shape(y_train)

    nb_tasks = y_train.shape[1]
    tasks_dims = []
    tasks_labels = []

    for task_idx in range(nb_tasks):
        output_dim, train_labels = _infer_single_task(y_train[:, task_idx], task_idx)
        tasks_dims.append(output_dim)
        tasks_labels.append(train_labels)

    return tasks_dims, tasks_labels
