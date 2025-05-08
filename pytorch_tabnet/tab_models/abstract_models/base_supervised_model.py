"""Supervised model base module for TabNet implementations.

This module provides the abstract base class for TabNet supervised models,
extending the TabModel with functionality specific to supervised learning tasks.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from pytorch_tabnet import tab_network
from pytorch_tabnet.data_handlers import PredictDataset, TBDataLoader, create_dataloaders
from pytorch_tabnet.metrics import MetricContainer, check_metrics
from pytorch_tabnet.tab_models.abstract_models import TabModel
from pytorch_tabnet.utils import create_group_matrix
from pytorch_tabnet.utils.matrices import _create_explain_matrix


@dataclass
class _TabSupervisedModel(TabModel):
    """Abstract base class for TabNet supervised models."""

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
