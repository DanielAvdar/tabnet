import pytest
import numpy as np
import scipy

from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor


@pytest.mark.parametrize(
    "model_params, fit_params, X_train, X_valid",
    [
        (
            dict(),
            dict(
                pretraining_ratio=0.8,
                max_epochs=1,
                batch_size=32,
                virtual_batch_size=16,
            ),
            np.random.rand(1000, 10),
            np.random.rand(50, 10),
        ),
        (
            dict(
                cat_idxs=[0, 1, 2, 3, 4],
                cat_dims=[5, 5, 5, 5, 5],
            ),
            dict(
                pretraining_ratio=0.8,
                max_epochs=1,
                batch_size=32,
                virtual_batch_size=32,
            ),
            np.random.rand(1000, 10),
            np.random.rand(50, 10),
        ),
        (
            dict(cat_idxs=[0, 1, 2, 3, 4], cat_dims=[5, 5, 5, 5, 5], cat_emb_dim=5),
            dict(
                pretraining_ratio=0.8,
                max_epochs=1,
                batch_size=16,
                virtual_batch_size=32,
                weights=np.ones(
                    100
                ),  # todo fix bug in TabNetPretrainer for 1000 samples
            ),
            np.random.rand(100, 10),
            np.random.rand(200, 10),
        ),
        (
            dict(cat_idxs=[0, 1, 2, 3, 4], cat_dims=[5, 5, 5, 5, 5], n_shared=1),
            dict(
                pretraining_ratio=0.8,
                max_epochs=1,
                batch_size=32,
                virtual_batch_size=16,
            ),
            scipy.sparse.csr_matrix((1000, 10)),
            scipy.sparse.csr_matrix((50, 10)),
        ),
    ],
)
@pytest.mark.parametrize("mask_type", ["sparsemax", "entmax"])
def test_pretrainer_fit(model_params, fit_params, X_train, X_valid, mask_type):
    """Test TabNetPretrainer fit method."""
    unsupervised_model = TabNetPretrainer(**model_params, mask_type=mask_type)
    unsupervised_model.fit(X_train=X_train, eval_set=[X_valid], **fit_params)
    assert hasattr(unsupervised_model, "network")
    assert unsupervised_model.network.pretraining_ratio == 0.8
    assert unsupervised_model.history.epoch_metrics
    unsupervised_model.save_model("test_model")
    unsupervised_model.load_model("test_model.zip")
    pred, _ = unsupervised_model.predict(X_valid)
    assert pred.shape[0] == X_valid.shape[0]
    assert not np.isnan(pred).any()
    # if not sparse
    if not scipy.sparse.issparse(X_train):
        tab_class = TabNetClassifier()
        tab_class.fit(
            X_train=X_train,
            y_train=np.random.randint(0, 2, size=X_train.shape[0]),
            from_unsupervised=unsupervised_model,
        )
        multi_tab_class = TabNetMultiTaskClassifier()
        multi_tab_class.fit(
            X_train=X_train,
            y_train=np.random.randint(0, 2, size=(X_train.shape[0], 2)),
            from_unsupervised=unsupervised_model,
        )
        tab_reg = TabNetRegressor()
        tab_reg.fit(
            X_train=X_train,
            y_train=np.random.rand(X_train.shape[0]).reshape(-1, 1),
            from_unsupervised=unsupervised_model,
        )
        multi_tab_reg = TabNetRegressor()
        multi_tab_reg.fit(
            X_train=X_train,
            y_train=np.random.rand(X_train.shape[0] * 3).reshape(-1, 3),
            from_unsupervised=unsupervised_model,
        )
