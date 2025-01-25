from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from pytorch_tabnet import tab_network
from pytorch_tabnet.abstract_model import TabModel
from pytorch_tabnet.data_handlers import PredictDataset, SparsePredictDataset, create_dataloaders_pt, validate_eval_set
from pytorch_tabnet.metrics import (
    UnsupervisedLoss,
    UnsupMetricContainer,
    check_metrics,
)
from pytorch_tabnet.utils import (
    check_input,
    create_explain_matrix,
    create_group_matrix,
    filter_weights,
)


@dataclass
class TabNetPretrainer(TabModel):
    def __post_init__(self) -> None:
        super(TabNetPretrainer, self).__post_init__()
        self._task = "unsupervised"
        self._default_loss = UnsupervisedLoss
        self._default_metric = "unsup_loss_numpy"

    def prepare_target(self, y: np.ndarray) -> np.ndarray:
        return y

    def compute_loss(
        self,
        output: torch.Tensor,
        embedded_x: torch.Tensor,
        obf_vars: torch.Tensor,
        w: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = self.loss_fn(output, embedded_x, obf_vars)
        if w is not None:
            loss = loss * w
        return loss

    def update_fit_params(  # type: ignore[override]
        self,
        weights: np.ndarray,
    ) -> None:
        # self.updated_weights = weights
        filter_weights(weights)
        self.preds_mapper = None

    def fit(  # type: ignore[override]
        self,
        X_train: np.ndarray,
        eval_set: Optional[List[Union[np.ndarray, List[np.ndarray]]]] = None,
        eval_name: Optional[List[str]] = None,
        loss_fn: Optional[Callable] = None,
        pretraining_ratio: float = 0.5,
        weights: Union[int, np.ndarray] = 0,
        max_epochs: int = 100,
        patience: int = 10,
        batch_size: int = 1024,
        virtual_batch_size: int = 128,
        num_workers: int = 0,
        drop_last: bool = True,
        callbacks: Optional[List[Callable]] = None,
        pin_memory: bool = True,
        warm_start: bool = False,
        *args: List,
        **kwargs: Dict,
    ) -> None:
        """Train a neural network stored in self.network
        Using train_dataloader for training data and
        valid_dataloader for validation.

        Parameters
        ----------
        X_train : np.ndarray
            Train set to reconstruct in self supervision
        eval_set : list of np.array
            List of evaluation set
            The last one is used for early stopping
        eval_name : list of str
            List of eval set names.
        eval_metric : list of str
            List of evaluation metrics.
            The last metric is used for early stopping.
        loss_fn : callable or None
            a PyTorch loss function
            should be left to None for self supervised and non experts
        pretraining_ratio : float
            Between 0 and 1, percentage of feature to mask for reconstruction
        weights : np.array
            Sampling weights for each example.
        max_epochs : int
            Maximum number of epochs during training
        patience : int
            Number of consecutive non improving epoch before early stopping
        batch_size : int
            Training batch size
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization (virtual_batch_size < batch_size)
        num_workers : int
            Number of workers used in torch.utils.data.DataLoader
        drop_last : bool
            Whether to drop last batch during training
        callbacks : list of callback function
            List of custom callbacks
        pin_memory: bool
            Whether to set pin_memory to True or False during training
        """
        # update model name

        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.input_dim = X_train.shape[1]
        self._stop_training = False
        self.pin_memory = pin_memory and (self.device.type != "cpu")
        self.pretraining_ratio = pretraining_ratio
        eval_set = eval_set if eval_set else []

        if loss_fn is None:
            self.loss_fn = self._default_loss
        else:
            self.loss_fn = loss_fn

        check_input(X_train)

        self.update_fit_params(
            weights,
        )

        # Validate and reformat eval set depending on training data
        eval_names = validate_eval_set(eval_set, eval_name, X_train)
        train_dataloader, valid_dataloaders = self._construct_loaders(X_train, eval_set, weights=weights)

        if not hasattr(self, "network") or not warm_start:
            # model has never been fitted before of warm_start is False
            self._set_network()

        self._update_network_params()
        self._set_metrics(eval_names)
        self._set_optimizer()
        self._set_callbacks(callbacks)

        # Call method on_train_begin for all callbacks
        self._callback_container.on_train_begin()

        # Training loop over epochs
        for epoch_idx in range(self.max_epochs):
            # Call method on_epoch_begin for all callbacks
            self._callback_container.on_epoch_begin(epoch_idx)

            self._train_epoch(train_dataloader)

            # Apply predict epoch to all eval sets
            for eval_name_, valid_dataloader in zip(eval_names, valid_dataloaders, strict=False):
                self._predict_epoch(eval_name_, valid_dataloader)

            # Call method on_epoch_end for all callbacks
            self._callback_container.on_epoch_end(epoch_idx, logs=self.history.epoch_metrics)

            if self._stop_training:
                break

        # Call method on_train_end for all callbacks
        self._callback_container.on_train_end()
        self.network.eval()

    def _set_network(self) -> None:
        """Setup the network and explain matrix."""
        if not hasattr(self, "pretraining_ratio"):
            self.pretraining_ratio = 0.5
        torch.manual_seed(self.seed)

        self.group_matrix = create_group_matrix(self.grouped_features, self.input_dim)

        self.network = tab_network.TabNetPretraining(
            self.input_dim,
            pretraining_ratio=self.pretraining_ratio,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            cat_emb_dim=self.cat_emb_dim,  # type: ignore
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            n_shared_decoder=self.n_shared_decoder,
            n_indep_decoder=self.n_indep_decoder,
            epsilon=self.epsilon,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            mask_type=self.mask_type,
            group_attention_matrix=self.group_matrix.to(self.device),
        ).to(self.device)
        if self.compile_backend in self.compile_backends:
            self.network = torch.compile(self.network, backend=self.compile_backend)
        self.reducing_matrix = create_explain_matrix(
            self.network.input_dim,
            self.network.cat_emb_dim,
            self.network.cat_idxs,
            self.network.post_embed_dim,
        )

    def _update_network_params(self) -> None:
        self.network.virtual_batch_size = self.virtual_batch_size
        self.network.pretraining_ratio = self.pretraining_ratio

    def _set_metrics(self, eval_names: List[str]) -> None:  # type: ignore[override]
        """Set attributes relative to the metrics.

        Parameters
        ----------
        metrics : list of str
            List of eval metric names.
        eval_names : list of str
            List of eval set names.

        """
        metrics = [self._default_metric]

        metrics = check_metrics(metrics)
        # Set metric container for each sets
        self._metric_container_dict = {}
        for name in eval_names:
            self._metric_container_dict.update({
                name: UnsupMetricContainer(metrics, prefix=f"{name}_")  # type: ignore
            })

        self._metrics = []
        self._metrics_names = []
        for _, metric_container in self._metric_container_dict.items():
            self._metrics.extend(metric_container.metrics)
            self._metrics_names.extend(metric_container.names)

        # Early stopping metric is the last eval metric
        self.early_stopping_metric = self._metrics_names[-1] if len(self._metrics_names) > 0 else None

    def _construct_loaders(  # type: ignore[override]
        self,
        X_train: np.ndarray,
        eval_set: List[Union[np.ndarray, List[np.ndarray]]],
        weights: Union[int, Dict, np.array],
    ) -> tuple[DataLoader, List[DataLoader]]:  # todo: replace loader
        """Generate dataloaders for unsupervised train and eval set.

        Parameters
        ----------
        X_train : np.array
            Train set.
        eval_set : list of tuple
            List of eval tuple set (X, y).

        Returns
        -------
        train_dataloader : `torch.utils.data.Dataloader`
            Training dataloader.
        valid_dataloaders : list of `torch.utils.data.Dataloader`
            List of validation dataloaders.

        """
        train_dataloader, valid_dataloaders = create_dataloaders_pt(
            X_train,
            eval_set,
            weights,
            self.batch_size,
            self.num_workers,
            self.drop_last,
            self.pin_memory,
        )
        return train_dataloader, valid_dataloaders

    def _train_epoch(self, train_loader: DataLoader) -> None:  # todo: replace loader
        """
        Trains one epoch of the network in self.network

        Parameters
        ----------
        train_loader : a :class: `torch.utils.data.Dataloader`
            DataLoader with train set
        """
        self.network.train()

        for batch_idx, (X, _, _) in enumerate(train_loader):  # todo: replace loader
            self._callback_container.on_batch_begin(batch_idx)
            X = X.to(self.device, non_blocking=True)

            batch_logs = self._train_batch(X)

            self._callback_container.on_batch_end(batch_idx, batch_logs)

        epoch_logs = {"lr": self._optimizer.param_groups[-1]["lr"]}
        self.history.epoch_metrics.update(epoch_logs)

        return

    def _train_batch(self, X: torch.Tensor, w: Optional[torch.Tensor] = None) -> dict:  # type: ignore[override]
        """
        Trains one batch of data

        Parameters
        ----------
        X : torch.Tensor
            Train matrix

        Returns
        -------
        batch_outs : dict
            Dictionnary with "y": target and "score": prediction scores.
        batch_logs : dict
            Dictionnary with "batch_size" and "loss".
        """
        batch_logs = {"batch_size": X.shape[0]}

        # X = X.to(self.device).float()

        for param in self.network.parameters():
            param.grad = None

        output, embedded_x, obf_vars = self.network(X)
        loss = self.compute_loss(output, embedded_x, obf_vars)

        # Perform backward pass and optimization
        loss.backward()
        if self.clip_value:
            clip_grad_norm_(self.network.parameters(), self.clip_value)
        self._optimizer.step()

        batch_logs["loss"] = loss.cpu().detach().numpy().item()

        return batch_logs

    def _predict_epoch(self, name: str, loader: DataLoader) -> None:  # todo: replace loader
        """
        Predict an epoch and update metrics.

        Parameters
        ----------
        name : str
            Name of the validation set
        loader : torch.utils.data.Dataloader
                DataLoader with validation set
        """
        # Setting network on evaluation mode
        self.network.eval()

        list_output = []
        list_embedded_x = []
        list_obfuscation = []
        # Main loop
        for _batch_idx, (X, _, _) in enumerate(loader):
            output, embedded_x, obf_vars = self._predict_batch(X)
            list_output.append(output)
            list_embedded_x.append(embedded_x)
            list_obfuscation.append(obf_vars)

        output, embedded_x, obf_vars = self.stack_batches(list_output, list_embedded_x, list_obfuscation)

        metrics_logs = self._metric_container_dict[name](output, embedded_x, obf_vars)
        self.network.train()
        self.history.epoch_metrics.update(metrics_logs)
        return

    def _predict_batch(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict one batch of data.

        Parameters
        ----------
        X : torch.Tensor
            Owned products

        Returns
        -------
        np.array
            model scores
        """
        X = X.to(self.device).float()
        return self.network(X)

    def stack_batches(  # type: ignore[override]
        self,
        list_output: List[torch.Tensor],
        list_embedded_x: List[torch.Tensor],
        list_obfuscation: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # output = np.vstack(list_output)
        # embedded_x = np.vstack(list_embedded_x)
        # obf_vars = np.vstack(list_obfuscation)
        output = torch.vstack(list_output)
        embedded_x = torch.vstack(list_embedded_x)
        obf_vars = torch.vstack(list_obfuscation)
        return output, embedded_x, obf_vars

    def predict(self, X: Union[np.ndarray, scipy.sparse.csr_matrix]) -> tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on a batch (valid)

        Parameters
        ----------
        X : a :tensor: `torch.Tensor` or matrix: `scipy.sparse.csr_matrix`
            Input data

        Returns
        -------
        predictions : np.array
            Predictions of the regression problem
        """
        self.network.eval()

        if scipy.sparse.issparse(X):
            dataloader = DataLoader(
                SparsePredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            dataloader = DataLoader(
                PredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )

        results = []
        embedded_res = []
        with torch.no_grad():
            for _batch_nb, data in enumerate(dataloader):
                data = data.to(self.device).float()
                output, embeded_x, _ = self.network(data)
                # predictions = output.cpu().detach().numpy()
                predictions = output
                results.append(predictions)
                # embedded_res.append(embeded_x.cpu().detach().numpy())
                embedded_res.append(embeded_x)
        # res_output = np.vstack(results)
        # embedded_inputs = np.vstack(embedded_res)
        res_output = torch.vstack(results).cpu().detach().numpy()

        embedded_inputs = torch.vstack(embedded_res).cpu().detach().numpy()
        return res_output, embedded_inputs
