"""Abstract model definitions for TabNet."""

import warnings
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

# from torch.utils.data import DataLoader
from pytorch_tabnet import tab_network
from pytorch_tabnet.abstract_model import TabModel

# from torch.utils.data import DataLoader
# from torch.utils.data import DataLoader
# from torch.utils.data import DataLoader
# from torch.utils.data import DataLoader
from pytorch_tabnet.data_handlers import PredictDataset, TBDataLoader, create_dataloaders
from pytorch_tabnet.metrics import MetricContainer, check_metrics

# from torch.utils.data import DataLoader
from pytorch_tabnet.utils import (
    check_input,
    create_group_matrix,
    validate_eval_set,
)
from pytorch_tabnet.utils.matrices import _create_explain_matrix


@dataclass
class TabSupervisedModel(TabModel):
    """Abstract base class for TabNet models."""

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

        check_input(X_train)

        self.update_fit_params(
            X_train,
            y_train,
            eval_set,
            weights,
        )

        eval_names, eval_set = validate_eval_set(eval_set, eval_name, X_train, y_train)

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

    def _construct_loaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: List[Tuple[np.ndarray, np.ndarray]],
        weights: Union[int, Dict, np.array],
    ) -> Tuple[TBDataLoader, List[TBDataLoader]]:
        """Generate dataloaders for train and eval set.

        Parameters
        ----------
        X_train : np.array
            Train set.
        y_train : np.array
            Train targets.
        eval_set : list of tuple
            List of eval tuple set (X, y).
        weights : int, dict, or np.array
            Sample weights.

        Returns
        -------
        train_dataloader : torch.utils.data.Dataloader
            Training dataloader.
        valid_dataloaders : list of torch.utils.data.Dataloader
            List of validation dataloaders.

        """
        y_train_mapped = self.prepare_target(y_train)
        for i, (X, y) in enumerate(eval_set):
            y_mapped = self.prepare_target(y)
            eval_set[i] = (X, y_mapped)

        train_dataloader, valid_dataloaders = create_dataloaders(
            X_train,
            y_train_mapped,
            eval_set,
            weights,
            self.batch_size,
            self.num_workers,
            self.drop_last,
            self.pin_memory,
        )
        return train_dataloader, valid_dataloaders

    def _train_epoch(self, train_loader: TBDataLoader) -> None:
        """Train one epoch of the network in self.network.

        Parameters
        ----------
        train_loader : TBDataLoader
            DataLoader with train set.

        """
        self.network.train()

        for batch_idx, (X, y, w) in enumerate(train_loader):  # type: ignore
            self._callback_container.on_batch_begin(batch_idx)
            X = X.to(self.device)  # type: ignore
            y = y.to(self.device)  # type: ignore
            if w is not None:  # type: ignore
                w = w.to(self.device)  # type: ignore

            batch_logs = self._train_batch(X, y, w)

            self._callback_container.on_batch_end(batch_idx, batch_logs)

        epoch_logs = {"lr": self._optimizer.param_groups[-1]["lr"]}
        self.history.epoch_metrics.update(epoch_logs)

        return

    def _set_network(self) -> None:
        """Set up the network and explain matrix."""
        torch.manual_seed(self.seed)

        self.group_matrix = create_group_matrix(self.grouped_features, self.input_dim)

        self.network = tab_network.TabNet(
            self.input_dim,
            self.output_dim,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            cat_emb_dim=self.cat_emb_dim,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            epsilon=self.epsilon,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            mask_type=self.mask_type,
            group_attention_matrix=self.group_matrix.to(self.device),
        ).to(self.device)
        if self.compile_backend in self.compile_backends:
            self.network = torch.compile(self.network, backend=self.compile_backend)

        self.reducing_matrix = _create_explain_matrix(
            self.network.input_dim,
            self.network.cat_emb_dim,
            self.network.cat_idxs,
            self.network.post_embed_dim,
        )

    def _predict_batch(self, X: torch.Tensor) -> torch.Tensor:
        """Predict one batch of data.

        Parameters
        ----------
        X : torch.Tensor
            Owned products.

        Returns
        -------
        torch.Tensor
            Model scores.

        """
        scores, _ = self.network(X)

        if isinstance(scores, list):
            scores = [x for x in scores]
        else:
            scores = scores

        return scores

    def _set_metrics(self, metrics: Union[None, List[str]], eval_names: List[str]) -> None:
        """Set attributes relative to the metrics.

        Parameters
        ----------
        metrics : list of str
            List of eval metric names.
        eval_names : list of str
            List of eval set names.

        """
        metrics = metrics or [self._default_metric]

        metrics = check_metrics(metrics)
        self._metric_container_dict: Dict[str, MetricContainer] = {}
        for name in eval_names:
            self._metric_container_dict.update({name: MetricContainer(metrics, prefix=f"{name}_")})

        self._metrics: List = []
        self._metrics_names: List[str] = []
        for _, metric_container in self._metric_container_dict.items():
            self._metrics.extend(metric_container.metrics)
            self._metrics_names.extend(metric_container.names)

        self.early_stopping_metric: Union[None, str] = self._metrics_names[-1] if len(self._metrics_names) > 0 else None

    def _train_batch(self, X: torch.Tensor, y: torch.Tensor, w: Optional[torch.Tensor] = None) -> Dict:
        """Train one batch of data.

        Parameters
        ----------
        X : torch.Tensor
            Train matrix.
        y : torch.Tensor
            Target matrix.
        w : Optional[torch.Tensor]
            Optional sample weights.

        Returns
        -------
        dict
            Dictionary with batch size and loss.

        """
        batch_logs = {"batch_size": X.shape[0]}

        for param in self.network.parameters():
            param.grad = None

        output, M_loss = self.network(X)

        loss = self.compute_loss(output, y, w)
        loss = loss - self.lambda_sparse * M_loss

        loss.backward()
        if self.clip_value:
            clip_grad_norm_(self.network.parameters(), self.clip_value)
        self._optimizer.step()

        batch_logs["loss"] = loss.item()

        return batch_logs

    def _predict_epoch(self, name: str, loader: TBDataLoader) -> None:
        """Predict an epoch and update metrics.

        Parameters
        ----------
        name : str
            Name of the validation set.
        loader : TBDataLoader
            DataLoader with validation set.

        """
        self.network.eval()

        list_y_true = []
        list_y_score = []
        list_w_ture = []
        with torch.no_grad():
            for _batch_idx, (X, y, w) in enumerate(loader):  # type: ignore
                scores = self._predict_batch(X.to(self.device, non_blocking=True).float())  # type: ignore
                list_y_true.append(y.to(self.device, non_blocking=True))  # type: ignore

                list_y_score.append(scores)
                if w is not None:  # type: ignore
                    list_w_ture.append(w.to(self.device, non_blocking=True))  # type: ignore
        w_true = None
        if list_w_ture:
            w_true = torch.cat(list_w_ture, dim=0)

        y_true, scores = self.stack_batches(list_y_true, list_y_score)

        metrics_logs = self._metric_container_dict[name](y_true, scores, w_true)
        self.network.train()
        self.history.epoch_metrics.update(metrics_logs)
        return

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on a batch (valid).

        Parameters
        ----------
        X : a :tensor: `torch.Tensor` or matrix: `scipy.sparse.csr_matrix`
            Input data.

        Returns
        -------
        np.ndarray
            Predictions of the regression problem.

        """
        self.network.eval()

        dataloader = TBDataLoader(
            name="predict",
            dataset=PredictDataset(X),
            batch_size=self.batch_size,
            predict=True,
        )

        results = []
        with torch.no_grad():
            for _batch_nb, (data, _, _) in enumerate(iter(dataloader)):  # type: ignore
                data = data.to(self.device, non_blocking=True).float()
                output, _M_loss = self.network(data)
                predictions = output.cpu().detach().numpy()
                results.append(predictions)
        res = np.vstack(results)
        return self.predict_func(res)

    @abstractmethod
    def prepare_target(self, y: np.ndarray) -> torch.Tensor:
        """Prepare target before training.

        Parameters
        ----------
        y : np.ndarray
            Target matrix.

        Returns
        -------
        torch.Tensor
            Converted target matrix.

        """
