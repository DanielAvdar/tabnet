import pytest
import numpy as np
from pytorch_tabnet.pretraining import TabNetPretrainer

# @pytest.fixture
# def X_train():
#     return np.random.rand(100, 10)
#
# @pytest.fixture
# def X_valid():
#     return np.random.rand(50, 10)
#
# @pytest.fixture
# def unsupervised_model():
#     return TabNetPretrainer()
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