import numpy as np
import pytest
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from scipy.sparse import csr_matrix


# @pytest.fixture
# def sample_data_regressor():
#     X_train = np.random.rand(100, 10)
#     y_train = np.random.rand(100, 3)
#     eval_set = [(np.random.rand(20, 10), np.random.rand(20, 3))]
#     return X_train, y_train, eval_set


@pytest.fixture(
    params=[
        (
            np.random.rand(100, 10),
            np.random.rand(100, 3),
            [(np.random.rand(20, 10), np.random.rand(20, 3))],
        ),
        (
            csr_matrix((100, 10)),
            np.random.rand(100, 3),
            [(csr_matrix((20, 10)), np.random.rand(20, 3))],
        ),
    ]
)
def sample_data_regressor(request):
    return request.param


@pytest.fixture
def regressor_instance():
    return TabNetRegressor()


def test_reg_fig(sample_data_regressor, regressor_instance):
    X_train, y_train, eval_set = sample_data_regressor
    regressor_instance.fit(X_train, y_train, eval_set=eval_set)
    assert regressor_instance.output_dim == 3
    assert regressor_instance._default_metric == "mse"
    pred = regressor_instance.predict(X_train)
    assert pred.shape[0] == X_train.shape[0]
    assert not np.isnan(pred).any()


def test_update_fit_params_regressor(sample_data_regressor, regressor_instance):
    X_train, y_train, eval_set = sample_data_regressor
    regressor_instance.update_fit_params(X_train, y_train, eval_set, weights=None)
    assert regressor_instance.output_dim == 3
    assert regressor_instance._default_metric == "mse"
    assert regressor_instance.updated_weights is None


def test_prepare_target_regressor(regressor_instance):
    y = np.array([0.1, 1.5, 0.3, 1.8])
    transformed_y = regressor_instance.prepare_target(y)
    assert np.array_equal(transformed_y, y)


def test_compute_loss_regressor(regressor_instance):
    y_true = torch.tensor([0.1, 1.5, 0.3, 1.8])
    y_pred = torch.tensor([0.2, 1.4, 0.4, 1.7])

    regressor_instance.loss_fn = regressor_instance._default_loss
    loss = regressor_instance.compute_loss(y_pred, y_true)
    assert loss.item() > 0


@pytest.mark.parametrize(
    "list_y_true, list_y_score, expected_y_true_shape, expected_y_score_shape",
    [
        (
            [np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([[0.5, 0.6], [0.7, 0.8]])],
            [np.array([[0.2, 0.3], [0.4, 0.5]]), np.array([[0.6, 0.7], [0.8, 0.9]])],
            (4, 2),
            (4, 2),
        ),
    ],
)
def test_stack_batches_regressor(
    regressor_instance,
    list_y_true,
    list_y_score,
    expected_y_true_shape,
    expected_y_score_shape,
):
    y_true, y_score = regressor_instance.stack_batches(list_y_true, list_y_score)
    assert y_true.shape == expected_y_true_shape
    assert y_score.shape == expected_y_score_shape


def test_predict_func_regressor(regressor_instance):
    outputs = np.array([[0.2, 0.3], [0.4, 0.5], [0.1, 0.9]])
    predictions = regressor_instance.predict_func(outputs)
    assert np.array_equal(predictions, outputs)
