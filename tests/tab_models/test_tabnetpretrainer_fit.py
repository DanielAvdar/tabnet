import pytest
import numpy as np
from pytorch_tabnet.pretraining import TabNetPretrainer


@pytest.mark.parametrize(
    "model_params, fit_params, X_train, X_valid",
    [
        (
        dict(),
        dict(pretraining_ratio=0.8, max_epochs=1, batch_size=32, virtual_batch_size=32),
        np.random.rand(100, 10),
        np.random.rand(50, 10)
        ),
        (
                dict(cat_idxs=[0, 1, 2, 3, 4], cat_dims=[5, 5, 5, 5, 5], ),
                dict(pretraining_ratio=0.8, max_epochs=1, batch_size=32, virtual_batch_size=32),
                np.random.rand(100, 10),
                np.random.rand(50, 10)
        ),
        (
        dict(cat_idxs=[0, 1, 2, 3, 4], cat_dims=[5, 5, 5, 5, 5],cat_emb_dim=5),
        dict(pretraining_ratio=0.8, max_epochs=1, batch_size=16, virtual_batch_size=32,weights=np.ones(100)),
        np.random.rand(100, 10),
        np.random.rand(200, 10)
)

,


    ]
)
def test_pretrainer_fit(model_params,fit_params, X_train, X_valid):
    """Test TabNetPretrainer fit method."""
    unsupervised_model = TabNetPretrainer(**model_params)
    unsupervised_model.fit(
        X_train=X_train,
        eval_set=[X_valid],
        **fit_params
    )
    assert hasattr(unsupervised_model, 'network')
    assert unsupervised_model.network.pretraining_ratio == 0.8
    assert unsupervised_model.history.epoch_metrics
    unsupervised_model.save_model("test_model")
    unsupervised_model.load_model("test_model.zip")
    pred,_ = unsupervised_model.predict(X_valid)
    assert pred.shape[0] == X_valid.shape[0]
