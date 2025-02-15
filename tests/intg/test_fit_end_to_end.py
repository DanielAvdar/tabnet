import numpy as np
import pytest
import scipy

from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor, MultiTabNetRegressor


# compile_backends = [
#     "no-compile",
# ]  # "onnxrt"]
#
# compile_backends += (
#     [
#         "onnxrt",
#     ]
#     if torch.onnx.is_onnxrt_backend_supported()
#     else []
# )
# compile_backends += ["cudagraphs"] if torch.cuda.is_available() else []


@pytest.mark.parametrize(
    "model_params, fit_params, X_train, X_valid",
    [
        (
            dict(),
            dict(
                pretraining_ratio=0.8,
                max_epochs=3,
                batch_size=128,
                # virtual_batch_size=16,
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
                max_epochs=3,
                batch_size=128,
                # virtual_batch_size=128,
            ),
            np.random.rand(1000, 10),
            np.random.rand(50, 10),
        ),
        (
            dict(cat_idxs=[0, 1, 2, 3, 4], cat_dims=[5, 5, 5, 5, 5], cat_emb_dim=5),
            dict(
                pretraining_ratio=0.8,
                max_epochs=3,
                batch_size=128,
                # virtual_batch_size=128,
                weights=np.ones(100),  # todo fix bug in TabNetPretrainer for 1000 samples
            ),
            np.random.rand(100, 10),
            np.random.rand(200, 10),
        ),
        (
            dict(cat_idxs=[0, 1, 2, 3, 4], cat_dims=[5, 5, 5, 5, 5], n_shared=1),
            dict(
                pretraining_ratio=0.8,
                max_epochs=3,
                batch_size=128,
                # virtual_batch_size=16,
            ),
            scipy.sparse.csr_matrix((1000, 10)),
            scipy.sparse.csr_matrix((50, 10)),
        ),
    ],
)
@pytest.mark.parametrize("mask_type", ["sparsemax", "entmax"])
# @pytest.mark.parametrize("compile_backend", compile_backends)
@pytest.mark.parametrize("pin_memory", [True, False])
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("final_model", [
    TabNetClassifier,
    TabNetMultiTaskClassifier,
    TabNetRegressor,
    MultiTabNetRegressor,


]
)
def test_pretrainer_fit(
    model_params,
    fit_params,
    X_train,
    X_valid,
    mask_type,
    pin_memory,
    num_workers,
    final_model,
):
    """Test TabNetPretrainer fit method."""
    unsupervised_model = TabNetPretrainer(
        **model_params,
        mask_type=mask_type,
    )
    unsupervised_model.fit(X_train=X_train, eval_set=[X_valid], **fit_params, pin_memory=pin_memory)
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
        # tab_class = TabNetClassifier()
        # tab_class.fit(
        #     y_train=np.random.randint(0, 2, size=X_train.shape[0]),
        #     X_train=X_train,
        #     from_unsupervised=unsupervised_model,
        #     **fit_params,
        # )
        # multi_tab_class = TabNetMultiTaskClassifier()
        # multi_tab_class.fit(
        #     y_train=np.random.randint(0, 2, size=(X_train.shape[0], 2)),
        #     X_train=X_train,
        #     from_unsupervised=unsupervised_model,
        #     **fit_params,
        # )
        # tab_reg = TabNetRegressor()
        # tab_reg.fit(
        #     y_train=np.random.rand(X_train.shape[0]).reshape(-1, 1),
        #     X_train=X_train,
        #     from_unsupervised=unsupervised_model,
        #     **fit_params,
        # )
        # multi_tab_reg = TabNetRegressor()
        # multi_tab_reg.fit(
        #     y_train=np.random.rand(X_train.shape[0] * 3).reshape(-1, 3),
        #     X_train=X_train,
        #     from_unsupervised=unsupervised_model,
        #     **fit_params,
        # )
        y_train= None
        if final_model == TabNetClassifier:
            y_train = np.random.randint(0, 2, size=X_train.shape[0])
        elif final_model == TabNetMultiTaskClassifier:
            y_train = np.random.randint(0, 2, size=(X_train.shape[0], 2))
        elif final_model == TabNetRegressor:
            y_train = np.random.rand(X_train.shape[0]).reshape(-1, 1)
        else:
            y_train = np.random.rand(X_train.shape[0] * 3).reshape(-1, 3)
        final_model_instance = final_model()
        final_model_instance.fit(
            y_train=y_train,
            X_train=X_train,
            from_unsupervised=unsupervised_model,
            **fit_params,
            # num_workers=num_workers,
        )
