import numpy as np
import pytest
import torch
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier


@pytest.fixture
def sample_data():
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, size=(100, 3))
    X_test = np.random.rand(50, 10)
    y_test = np.random.randint(0, 2, size=(50, 3))
    return X_train, y_train, X_test, y_test


@pytest.fixture
def classifier():
    return TabNetMultiTaskClassifier()


def test_update_fit_params(sample_data, classifier):
    X_train, y_train, X_test, y_test = sample_data
    classifier.update_fit_params(X_train, y_train, [(X_test, y_test)], None)

    assert hasattr(classifier, "output_dim")
    assert hasattr(classifier, "classes_")
    assert hasattr(classifier, "target_mapper")
    assert hasattr(classifier, "updated_weights")


def test_prepare_target(sample_data, classifier):
    _, y_train, _, _ = sample_data
    classifier.target_mapper = [
        {0: 0, 1: 1},
        {0: 0, 1: 1},
        {0: 0, 1: 1},
    ]
    y_mapped = classifier.prepare_target(y_train)

    assert y_mapped.shape == y_train.shape
    assert np.all(np.isin(y_mapped, [0, 1]))


def test_compute_loss(sample_data, classifier):
    _, y_train, _, _ = sample_data
    classifier.loss_fn = torch.nn.CrossEntropyLoss()
    y_pred = [torch.rand(size=(100, 2)) for _ in range(3)]
    y_true = torch.tensor(y_train)

    loss = classifier.compute_loss(y_pred, y_true)

    assert isinstance(loss, torch.Tensor)
    assert loss > 0


def test_stack_batches(sample_data,classifier):
    classifier = TabNetMultiTaskClassifier()
    X_train, y_train, X_test, y_test = sample_data
    classifier.fit(X_train, y_train,  max_epochs=1,eval_set=[(X_test, y_test)])
    y_true_list = [np.random.randint(0, 2, size=(20, 3))]
    y_pred_list = [np.random.rand(20, 3) for _ in range(3)]
    y_true, y_score = classifier.stack_batches(y_true_list, [y_pred_list])

    assert y_true.shape[0] == 20
    assert len(y_score) == 3
    for score in y_score:
        assert np.allclose(np.sum(score, axis=1), 1)


def test_predict(sample_data, classifier):
    X_train, y_train, X_test, y_test = sample_data
    classifier.fit(X_train, y_train,  max_epochs=1,eval_set=[(X_test, y_test)])
    predictions = classifier.predict(X_test)

    assert isinstance(predictions, list)
    assert len(predictions) == 3
    for task_prediction in predictions:
        assert task_prediction.shape[0] == X_test.shape[0]


def test_predict_proba(sample_data, classifier):
    X_train, y_train, X_test, y_test = sample_data
    classifier.fit(X_train, y_train,  max_epochs=1,eval_set=[(X_test, y_test)], weights=np.ones(100))
    probabilities = classifier.predict_proba(X_test)

    assert isinstance(probabilities, list)
    assert len(probabilities) == 3
    for task_proba in probabilities:
        assert task_proba.shape[0] == X_test.shape[0]
        assert np.allclose(np.sum(task_proba, axis=1), 1)