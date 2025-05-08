"""Abstract supervised model definition for TabNet."""

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from sklearn.utils import check_array

from ...utils import (
    validate_eval_set,
)
from . import TabModel
from .base_supervised_model import _TabSupervisedModel


@dataclass
class TabSupervisedModel(_TabSupervisedModel):
    """Abstract base class for TabNet supervised models."""

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: Union[None, List[Tuple[np.ndarray, np.ndarray]]] = None,
        eval_name: Union[None, List[str]] = None,
        eval_metric: Union[None, List[str]] = None,
        loss_fn: Union[None, Callable] = None,
        weights: Union[int, Dict, np.array] = 0,
        max_epochs: int = 100,
        patience: int = 10,
        batch_size: int = 1024,
        virtual_batch_size: int = None,
        num_workers: int = 0,
        drop_last: bool = True,
        callbacks: Union[None, List] = None,
        pin_memory: bool = True,
        from_unsupervised: Union[None, "TabModel"] = None,
        warm_start: bool = False,
        compute_importance: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Train a neural network stored in self.network.

        Uses train_dataloader for training data and valid_dataloader for validation.

        Parameters
        ----------
        X_train : np.ndarray
            Train set.
        y_train : np.array
            Train targets.
        eval_set : list of tuple
            List of eval tuple set (X, y). The last one is used for early stopping.
        eval_name : list of str
            List of eval set names.
        eval_metric : list of str
            List of evaluation metrics. The last metric is used for early stopping.
        loss_fn : callable or None
            PyTorch loss function.
        weights : bool or dict
            0 for no balancing, 1 for automated balancing, dict for custom weights per class.
        max_epochs : int
            Maximum number of epochs during training.
        patience : int
            Number of consecutive non-improving epochs before early stopping.
        batch_size : int
            Training batch size.
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization (virtual_batch_size < batch_size).
        num_workers : int
            Number of workers used in torch.utils.data.DataLoader.
        drop_last : bool
            Whether to drop last batch during training.
        callbacks : list of callback function
            List of custom callbacks.
        pin_memory: bool
            Whether to set pin_memory to True or False during training.
        from_unsupervised: unsupervised trained model
            Use a previously self-supervised model as starting weights.
        warm_start: bool
            If True, current model parameters are used to start training.
        compute_importance : bool
            Whether to compute feature importance.
        augmentations : callable or None
            Data augmentation function.
        *args : list
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        """
        self.max_epochs: int = max_epochs
        self.patience: int = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size or batch_size
        self.num_workers: int = num_workers
        self.drop_last: bool = drop_last
        self.input_dim: int = X_train.shape[1]
        self._stop_training: bool = False
        self.pin_memory: bool = pin_memory and (self.device.type != "cpu")
        self.compute_importance: bool = compute_importance

        eval_set = eval_set if eval_set else []

        if loss_fn is None:
            self.loss_fn = self._default_loss
        else:
            self.loss_fn = loss_fn

        check_array(X_train)

        self.update_fit_params(
            X_train,
            y_train,
            eval_set,
            weights,
        )
        eval_names = eval_name or [f"val_{i}" for i in range(len(eval_set))]

        validate_eval_set(eval_set, eval_names, X_train, y_train)

        train_dataloader, valid_dataloaders = self._construct_loaders(
            X_train,
            y_train,
            eval_set,
            self.weight_updater(weights=weights),
        )

        if from_unsupervised is not None:
            self.__update__(**from_unsupervised.get_params())

        if not hasattr(self, "network") or not warm_start:
            self._set_network()
        self._update_network_params()
        self._set_metrics(eval_metric, eval_names)
        self._set_optimizer()
        self._set_callbacks(callbacks)

        if from_unsupervised is not None:
            self.load_weights_from_unsupervised(from_unsupervised)
            warnings.warn("Loading weights from unsupervised pretraining", stacklevel=2)
        self._callback_container.on_train_begin()

        for epoch_idx in range(self.max_epochs):
            self._callback_container.on_epoch_begin(epoch_idx)
            self._train_epoch(train_dataloader)
            for eval_name_, valid_dataloader in zip(eval_names, valid_dataloaders, strict=False):
                self._predict_epoch(eval_name_, valid_dataloader)
            self._callback_container.on_epoch_end(epoch_idx, logs=self.history.epoch_metrics)

            if self._stop_training:
                break

        self._callback_container.on_train_end()
        self.network.eval()

        if self.compute_importance:
            self.feature_importances_ = self._compute_feature_importances(X_train)
