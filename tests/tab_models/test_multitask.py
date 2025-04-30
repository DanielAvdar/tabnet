import numpy as np
import pytest
import torch

from pytorch_tabnet.multitask import TabNetMultiTaskClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_torch(X):
    return torch.tensor(X, device=device)


@pytest.fixture(
    params=[
        (
            np.random.rand(1000, 10),
            np.random.randint(0, 2, size=(1000, 3)),
            np.random.rand(50, 10),
            np.random.randint(0, 2, size=(50, 3)),
        ),
        # (csr_matrix((1000, 10)), np.random.randint(0, 2, size=(1000, 3)), csr_matrix((50, 10)), np.random.randint(0, 2, size=(50, 3))),
        # (csr_matrix((1000, 10)), csr_array((1000, 3)), csr_matrix((50, 10)), csr_array((50, 3))), not implemented
    ]
)
def sample_data(request):
    return request.param


@pytest.fixture
def classifier():
    return TabNetMultiTaskClassifier()


def test_update_fit_params(sample_data, classifier):
    X_train, y_train, X_test, y_test = sample_data
    classifier.update_fit_params(X_train, y_train, [(X_test, y_test)], None)

    assert hasattr(classifier, "output_dim")
    assert hasattr(classifier, "classes_")
    assert hasattr(classifier, "target_mapper")
    # assert hasattr(classifier, "updated_weights")


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
    y_pred = [torch.rand(size=(1000, 2)) for _ in range(3)]
    y_true = torch.tensor(y_train)

    loss = classifier.compute_loss(y_pred, y_true)

    assert isinstance(loss, torch.Tensor)
    assert loss > 0


@pytest.mark.parametrize(
    "fit_params",
    [
        dict(max_epochs=1, batch_size=128, virtual_batch_size=128),
        dict(
            max_epochs=1,
            batch_size=128,
            virtual_batch_size=128,
            weights=np.ones(1000) / 1000,
        ),
        dict(max_epochs=1, batch_size=128, virtual_batch_size=128, eval_metric=["auc"]),
        dict(
            patience=1,
            batch_size=128,
            virtual_batch_size=128,
        ),
    ],
)
def test_class(sample_data, classifier, fit_params):
    X_train, y_train, X_test, y_test = sample_data

    classifier.fit(X_train, y_train, eval_set=[(X_test, y_test)], **fit_params)
    classifier._compute_feature_importances(X=X_train)

    X_train, y_train, X_test, y_test = (
        to_torch(X_train),
        to_torch(y_train),
        to_torch(X_test),
        to_torch(y_test),
    )

    probabilities = classifier.predict_proba(X_test)

    assert isinstance(probabilities, list)
    assert len(probabilities) == 3
    for task_proba in probabilities:
        assert task_proba.shape[0] == X_test.shape[0]
        assert np.allclose(np.sum(task_proba, axis=1), 1)

    y_true_list = [torch.randint(0, 2, size=(20, 3))]
    y_pred_list = [torch.rand(20, 3) for _ in range(3)]
    y_true, y_score = classifier.stack_batches(y_true_list, [y_pred_list])

    assert y_true.shape[0] == 20
    assert len(y_score) == 3
    for score in y_score:
        assert torch.allclose(torch.sum(score, dim=1), torch.tensor(1.0))
    predictions = classifier.predict(X_test)

    assert not all([np.isnan(p.astype(int)).any() for p in predictions])
    assert isinstance(predictions, list)
    assert len(predictions) == 3
    for task_prediction in predictions:
        assert task_prediction.shape[0] == X_test.shape[0]
