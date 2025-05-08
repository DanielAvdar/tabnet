"""Abstract model base module for TabNet implementations.

This module provides the abstract base class for TabNet models, defining common
functionality and interface across all TabNet implementations.
"""

import warnings
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

from pytorch_tabnet.callbacks import Callback, CallbackContainer, EarlyStopping, History, LRSchedulerCallback
from pytorch_tabnet.tab_models.abstract_models.base_model import _TabModel


@dataclass
class TabModel(_TabModel):
    """Abstract base class for TabNet models."""

    def _set_callbacks(self, custom_callbacks: Union[None, List]) -> None:
        """Set up the callbacks functions.

        Parameters
        ----------
        custom_callbacks : list of func
            List of callback functions.

        """
        callbacks: List[Union[History, EarlyStopping, LRSchedulerCallback, Callback]] = []
        self.history = History(self, verbose=self.verbose)
        callbacks.append(self.history)
        if (self.early_stopping_metric is not None) and (self.patience > 0):
            early_stopping = EarlyStopping(
                early_stopping_metric=self.early_stopping_metric,
                is_maximize=(self._metrics[-1]._maximize if len(self._metrics) > 0 else None),
                patience=self.patience,
            )
            callbacks.append(early_stopping)
        else:
            wrn_msg = "No early stopping will be performed, last training weights will be used."
            warnings.warn(wrn_msg, stacklevel=2)

        if self.scheduler_fn is not None:
            is_batch_level = self.scheduler_params.pop("is_batch_level", False)
            scheduler = LRSchedulerCallback(
                scheduler_fn=self.scheduler_fn,
                scheduler_params=self.scheduler_params,
                optimizer=self._optimizer,
                early_stopping_metric=self.early_stopping_metric,
                is_batch_level=is_batch_level,
            )
            callbacks.append(scheduler)

        if custom_callbacks:
            callbacks.extend(custom_callbacks)
        self._callback_container = CallbackContainer(callbacks)
        self._callback_container.set_trainer(self)

    def _set_optimizer(self) -> None:
        """Set up optimizer."""
        self._optimizer = self.optimizer_fn(self.network.parameters(), **self.optimizer_params)

    def _compute_feature_importances(self, X: np.ndarray) -> np.ndarray:
        """Compute global feature importance.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Feature importances.

        """
        M_explain, _ = self.explain(X, normalize=False)
        sum_explain = M_explain.sum(axis=0)
        feature_importances_ = sum_explain / np.sum(sum_explain)
        return feature_importances_

    def _update_network_params(self) -> None:
        """Update network parameters."""
        self.network.virtual_batch_size = self.virtual_batch_size

    def weight_updater(self, weights: Union[bool, Dict[Union[str, int], Any], Any]) -> Union[bool, Dict[Union[str, int], Any]]:
        """Update class weights for training.

        Parameters
        ----------
        weights : bool, dict, or any
            Class weights or indicator.

        Returns
        -------
        bool or dict
            Updated weights.

        """
        return weights

    @abstractmethod
    def update_fit_params(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: List[Tuple[np.ndarray, np.ndarray]],
        weights: Union[int, Dict],
    ) -> None:
        """Set attributes relative to fit function.

        Parameters
        ----------
        X_train : np.ndarray
            Train set.
        y_train : np.array
            Train targets.
        eval_set : list of tuple
            List of eval tuple set (X, y).
        weights : bool or dict
            0 for no balancing, 1 for automated balancing.

        """

    @abstractmethod
    def compute_loss(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Compute the loss.

        Parameters
        ----------
        *args : Any
            Arguments for loss computation.
        **kwargs : Any
            Keyword arguments for loss computation.

        Returns
        -------
        torch.Tensor
            Loss value.

        """

    @abstractmethod
    def predict_func(self, y_score: np.ndarray) -> np.ndarray:
        """Convert model scores to predictions.

        Parameters
        ----------
        y_score : np.ndarray
            Model scores.

        Returns
        -------
        np.ndarray
            Model predictions.

        """

    def stack_batches(self, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stack batches of predictions and targets.

        Parameters
        ----------
        *args : Any
            List of batch outputs.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Stacked predictions and targets.

        """
