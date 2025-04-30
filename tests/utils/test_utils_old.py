import numpy as np
import pytest
import torch

from pytorch_tabnet.utils import (
    create_group_matrix,
    validate_eval_set,
)
from pytorch_tabnet.utils.matrices import check_list_groups


@pytest.mark.parametrize(
    "eval_set, eval_name, error_message",
    [
        (
            [(np.random.rand(20, 15), np.random.rand(20))],
            ["validation"],
            "Number of columns is different between X_validation .* and X_train .*",
        ),
        (
            [(np.random.rand(20, 10), np.random.rand(20, 2))],
            ["validation"],
            "Dimension mismatch between y_validation .* and y_train .*",
        ),
        (
            [(np.random.rand(25, 10), np.random.rand(20))],
            ["validation"],
            "You need the same number of rows between X_validation .* and y_validation .*",
        ),
        (
            [(np.random.rand(20, 10), np.random.rand(20))],
            ["validation", "extra_name"],
            "eval_set and eval_name have not the same length",
        ),
        (
            [(np.random.rand(20, 10))],
            ["validation"],
            "Each tuple of eval_set need to have two elements",
        ),
    ],
)
def test_validate_eval_set_errors(eval_set, eval_name, error_message):
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)

    with pytest.raises(AssertionError, match=error_message):
        validate_eval_set(eval_set, eval_name, X_train, y_train)


@pytest.mark.parametrize("eval_name, expected_names", [(["validation"], ["validation"]), (None, ["val_0"])])
def test_validate_eval_set_valid_inputs_and_default_eval_name(eval_name, expected_names):
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)
    eval_set = [(np.random.rand(20, 10), np.random.rand(20))]

    validated_names, validated_set = validate_eval_set(eval_set, eval_name, X_train, y_train)

    assert validated_names == expected_names
    assert len(validated_set) == len(eval_set)
    assert np.array_equal(validated_set[0][0], eval_set[0][0])
    assert np.array_equal(validated_set[0][1], eval_set[0][1])


@pytest.mark.parametrize(
    "list_groups, input_dim, error_message",
    [
        # Check for non-list input
        (None, 10, "list_groups must be a list of list."),
        # Check for empty group
        (
            [[1], [], [3]],
            10,
            "Empty groups are forbidding please remove empty groups \\[\\]",
        ),
        # Check for duplicate group members
        (
            [[1, 2], [2, 3]],
            10,
            "One feature can only appear in one group, please check your grouped_features.",
        ),
        # Check for feature index out of bounds
        ([[1], [9, 11]], 10, "Number of features is 10 but one group contains 11."),
        # Check for invalid input type inside groups
        (
            [1, [2, 3]],
            10,
            "Groups must be given as a list of list, but found 1 in position 0.",
        ),
    ],
)
def test_check_list_groups_errors(list_groups, input_dim, error_message):
    with pytest.raises(AssertionError, match=error_message):
        check_list_groups(list_groups, input_dim)


@pytest.mark.parametrize(
    "list_groups, input_dim",
    [
        # Valid lists with no errors
        ([[1, 2, 3]], 10),
        ([[0], [1, 2], [3]], 5),
        ([], 10),
        ([[4, 5], [6]], 7),
    ],
)
def test_check_list_groups_valid(list_groups, input_dim):
    # Should not raise any errors
    check_list_groups(list_groups, input_dim)


@pytest.mark.parametrize(
    "list_groups, input_dim, expected_shape",
    [
        ([], 5, (5, 5)),
        ([[0, 1]], 4, (3, 4)),
        ([[0, 1], [2, 3]], 6, (4, 6)),
        ([[0], [2]], 3, (3, 3)),
    ],
)
def test_create_group_matrix_shape(list_groups, input_dim, expected_shape):
    result = create_group_matrix(list_groups, input_dim)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"


@pytest.mark.parametrize(
    "list_groups, input_dim, expected_sum",
    [
        ([[0, 1], [2]], 4, [1, 1, 1]),
        ([[0]], 3, [1, 1, 1]),
        ([], 3, [1, 1, 1]),
        ([[0, 1]], 5, [1, 1, 1, 1]),
    ],
)
def test_create_group_matrix_row_sum(list_groups, input_dim, expected_sum):
    result = create_group_matrix(list_groups, input_dim)
    row_sums = torch.sum(result, axis=1).numpy().tolist()
    assert row_sums == expected_sum, f"Expected row sums {expected_sum}, got {row_sums}"


@pytest.mark.parametrize(
    "list_groups, input_dim",
    [
        ([[0, 1], [1, 2]], 5),
        ([[0, 1], [0]], 5),
        ([[5], [0]], 5),
    ],
)
def test_create_group_matrix_invalid_input(list_groups, input_dim):
    with pytest.raises(AssertionError):
        create_group_matrix(list_groups, input_dim)


@pytest.mark.parametrize(
    "list_groups, input_dim, expected_matrix",
    [
        (
            [[0, 1]],
            3,
            torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
        ),
        (
            [],
            2,
            torch.eye(2, dtype=torch.float32),
        ),
        (
            [[0], [1]],
            3,
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
        ),
    ],
)
def test_create_group_matrix_correct_output(list_groups, input_dim, expected_matrix):
    result = create_group_matrix(list_groups, input_dim)
    assert torch.allclose(result, expected_matrix), f"Expected matrix {expected_matrix}, got {result}"
