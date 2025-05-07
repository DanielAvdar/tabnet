import numpy as np
import pytest

from pytorch_tabnet.tab_models.multitask import TabNetMultiTaskClassifier
from pytorch_tabnet.tab_models.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_models.tab_model import TabNetClassifier, TabNetRegressor


@pytest.mark.parametrize(
    " X_train, X_valid,weights,model_params, fit_params",
    [
        (
            np.random.rand(1000, 10),
            np.random.rand(50, 10),
            0,
            dict(),
            dict(
                pretraining_ratio=0.8,
                max_epochs=2,
                batch_size=128,
                # virtual_batch_size=16,
            ),
        ),
        (
            np.random.rand(50, 10),
            np.random.rand(1000, 10),
            0,
            dict(
                cat_idxs=[0, 1, 2, 3, 4],
                cat_dims=[5, 5, 5, 5, 5],
            ),
            dict(
                pretraining_ratio=0.8,
                max_epochs=2,
                batch_size=128,
                # virtual_batch_size=128,
            ),
        ),
        (
            np.random.rand(100, 10),
            np.random.rand(200, 10),
            np.ones(100),
            dict(cat_idxs=[0, 1, 2, 3, 4], cat_dims=[5, 5, 5, 5, 5], cat_emb_dim=5),
            dict(
                pretraining_ratio=0.8,
                max_epochs=2,
                batch_size=128,
            ),
        ),
    ],
)
@pytest.mark.parametrize("mask_type", ["sparsemax", "entmax"])
def test_pretrainer_fit(
    X_train,
    X_valid,
    weights,
    model_params,
    fit_params,
    mask_type,
):
    """Test TabNetPretrainer fit method."""
    unsupervised_model = TabNetPretrainer(
        **model_params,
        mask_type=mask_type,
    )
    unsupervised_model.fit(
        X_train=X_train,
        eval_set=[X_valid],
        weights=weights,
        **fit_params,
    )
    assert hasattr(unsupervised_model, "network")
    assert unsupervised_model.network.pretraining_ratio == 0.8
    assert unsupervised_model.history.epoch_metrics
    unsupervised_model.save_model("test_model")
    unsupervised_model.load_model("test_model.zip")
    pred, _ = unsupervised_model.predict(X_valid)
    assert pred.shape[0] == X_valid.shape[0]
    assert not np.isnan(pred).any()
    tab_class = TabNetClassifier()
    tab_class.fit(
        y_train=np.random.randint(0, 2, size=X_train.shape[0]),
        X_train=X_train,
        weights=weights,
        from_unsupervised=unsupervised_model,
        **fit_params,
    )
    multi_tab_class = TabNetMultiTaskClassifier()
    multi_tab_class.fit(
        y_train=np.random.randint(0, 2, size=(X_train.shape[0], 2)),
        X_train=X_train,
        weights=weights,
        from_unsupervised=unsupervised_model,
        **fit_params,
    )
    tab_reg = TabNetRegressor()
    tab_reg.fit(
        y_train=np.random.rand(X_train.shape[0]).reshape(-1, 1),
        X_train=X_train,
        weights=weights,
        from_unsupervised=unsupervised_model,
        **fit_params,
    )
    multi_tab_reg = TabNetRegressor()
    multi_tab_reg.fit(
        y_train=np.random.rand(X_train.shape[0] * 3).reshape(-1, 3),
        X_train=X_train,
        weights=weights,
        from_unsupervised=unsupervised_model,
        **fit_params,
    )
