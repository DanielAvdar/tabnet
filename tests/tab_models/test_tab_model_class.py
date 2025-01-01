import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix, csr_array
from pytorch_tabnet.tab_model import TabNetClassifier


# @pytest.fixture
# def sample_data():
#     X_train = np.random.rand(100, 10)
#     y_train = np.random.randint(0, 2, size=100)
#     eval_set = [(np.random.rand(20, 10), np.random.randint(0, 2, size=20))]
#     return X_train, y_train, eval_set

@pytest.fixture(
    params=[
        (np.random.rand(100, 10), np.random.randint(0, 2, size=100), [(np.random.rand(20, 10), np.random.randint(0, 2, size=20))]),
        (csr_matrix((100, 10)), np.random.randint(0, 2, size=100), [(csr_matrix((20, 10)), np.random.randint(0, 2, size=20))]),
    ]
)
def sample_data(request):
    return request.param



@pytest.fixture
def classifier_instance():
    return TabNetClassifier()


def test_class_fit(sample_data, classifier_instance):
    X_train, y_train, eval_set = sample_data
    weights = {0: 0.5, 1: 0.5}
    classifier_instance.fit(X_train, y_train, eval_set, weights=weights)
    assert classifier_instance.output_dim == 2 if isinstance(y_train, np.ndarray) else 1
    assert classifier_instance._default_metric == "auc"
    assert classifier_instance.classes_ is not None
    assert classifier_instance.updated_weights == {0: 0.5, 1: 0.5}
    proba_pred = classifier_instance.predict_proba(X_train)
    assert proba_pred.shape[0] == X_train.shape[0]

def test_update_fit_params(sample_data, classifier_instance):
    X_train, y_train, eval_set = sample_data
    weights = {0: 0.5, 1: 0.5}
    classifier_instance.update_fit_params(X_train, y_train, eval_set, weights)
    assert classifier_instance.output_dim == 2
    assert classifier_instance._default_metric == "auc"
    assert classifier_instance.classes_ is not None
    assert classifier_instance.updated_weights == {0: 0.5, 1: 0.5}

def test_weight_updater(classifier_instance):
    weights = {0: 0.7, 1: 0.3}
    classifier_instance.target_mapper = {0: 1, 1: 0}
    updated_weights = classifier_instance.weight_updater(weights)
    assert updated_weights == {1: 0.7, 0: 0.3}


def test_prepare_target(classifier_instance):
    classifier_instance.target_mapper = {0: 1, 1: 0}
    y = np.array([0, 1, 0, 1])
    transformed_y = classifier_instance.prepare_target(y)
    assert np.array_equal(transformed_y, np.array([1, 0, 1, 0]))


def test_compute_loss(classifier_instance):
    y_true = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    y_pred = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    classifier_instance.loss_fn =classifier_instance._default_loss
    loss = classifier_instance.compute_loss(y_pred, y_true)
    assert loss.item() > 0


@pytest.mark.parametrize(
    "list_y_true, list_y_score, expected_y_true, expected_y_score_shape",
    [
        ([np.array([0, 1]), np.array([1, 0])], 
         [np.array([[0.8, 0.2], [0.1, 0.9]]), np.array([[0.6, 0.4], [0.7, 0.3]])], 
         np.array([0, 1, 1, 0]), 
         (4, 2)),
    ]
)
def test_stack_batches(classifier_instance, list_y_true, list_y_score, expected_y_true, expected_y_score_shape):
    y_true, y_score = classifier_instance.stack_batches(list_y_true, list_y_score)
    assert np.array_equal(y_true, expected_y_true)
    assert y_score.shape == expected_y_score_shape


def test_predict_func(classifier_instance):
    classifier_instance.preds_mapper = {"0": 0, "1": 1}
    outputs = np.array([[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])
    predictions = classifier_instance.predict_func(outputs)
    assert np.array_equal(predictions, np.array([1, 0, 1]))


