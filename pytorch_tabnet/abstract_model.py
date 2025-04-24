"""Abstract model definitions for TabNet."""

import copy
import io
import json
import shutil
import warnings
import zipfile
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy
import torch
from scipy.sparse import csc_matrix
from sklearn.base import BaseEstimator
from torch.nn.utils import clip_grad_norm_

# from torch.utils.data import DataLoader
from pytorch_tabnet import tab_network
from pytorch_tabnet.callbacks import (
    Callback,
    CallbackContainer,
    EarlyStopping,
    History,
    LRSchedulerCallback,
)
from pytorch_tabnet.data_handlers import PredictDataset, SparsePredictDataset, TBDataLoader, create_dataloaders
from pytorch_tabnet.metrics import MetricContainer, check_metrics
from pytorch_tabnet.utils import (
    ComplexEncoder,
    check_embedding_parameters,
    check_input,
    check_warm_start,
    create_explain_matrix,
    create_group_matrix,
    define_device,
    validate_eval_set,
)


@dataclass
class TabModel(BaseEstimator):
    """Abstract base class for TabNet models."""

    compile_backends = ["inductor", "cudagraphs", "ipex", "onnxrt"]

    n_d: int = 8
    n_a: int = 8
    n_steps: int = 3
    gamma: float = 1.3
    cat_idxs: List[int] = field(default_factory=list)
    cat_dims: List[int] = field(default_factory=list)
    cat_emb_dim: Union[int, List[int]] = 1
    n_independent: int = 2
    n_shared: int = 2
    epsilon: float = 1e-15
    momentum: float = 0.02
    lambda_sparse: float = 1e-3
    seed: int = 0
    clip_value: int = 1
    verbose: int = 1
    optimizer_fn: Any = torch.optim.Adam
    optimizer_params: Dict = field(default_factory=lambda: dict(lr=2e-2))
    scheduler_fn: Any = None
    scheduler_params: Dict = field(default_factory=dict)
    mask_type: str = "sparsemax"
    input_dim: int = None
    output_dim: Union[List[int], int] = None
    device_name: str = "auto"
    n_shared_decoder: int = 1
    n_indep_decoder: int = 1
    grouped_features: List[List[int]] = field(default_factory=list)
    _callback_container: CallbackContainer = field(default=None, repr=False, init=False, compare=False)
    compile_backend: str = ""

    def __post_init__(self) -> None:
        """Initialize default values and device for TabModel."""
        self.batch_size: int = 1024
        self.virtual_batch_size: int = 128

        torch.manual_seed(self.seed)
        self.device = torch.device(define_device(self.device_name))
        if self.verbose != 0:
            warnings.warn(f"Device used : {self.device}", stacklevel=2)

        self.optimizer_fn = copy.deepcopy(self.optimizer_fn)
        self.scheduler_fn = copy.deepcopy(self.scheduler_fn)

        updated_params = check_embedding_parameters(self.cat_dims, self.cat_idxs, self.cat_emb_dim)
        self.cat_dims, self.cat_idxs, self.cat_emb_dim = updated_params

    def __update__(self, **kwargs: Any) -> None:
        """Update model parameters with provided keyword arguments.

        If a parameter does not already exist, it is created. Otherwise, it is overwritten with a warning.
        """
        update_list = [
            "cat_dims",
            "cat_emb_dim",
            "cat_idxs",
            "input_dim",
            "mask_type",
            "n_a",
            "n_d",
            "n_independent",
            "n_shared",
            "n_steps",
            "grouped_features",
        ]
        for var_name, value in kwargs.items():
            if var_name in update_list:
                try:
                    exec(f"global previous_val; previous_val = self.{var_name}")
                    if previous_val != value:  # type: ignore # noqa
                        wrn_msg = f"Pretraining: {var_name} changed from {previous_val} to {value}"  # type: ignore # noqa
                        warnings.warn(wrn_msg, stacklevel=2)
                        exec(f"self.{var_name} = value")
                except AttributeError:
                    exec(f"self.{var_name} = value")

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
        augmentations: Union[None, Any] = None,
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
        self.augmentations = augmentations
        self.compute_importance: bool = compute_importance

        if self.augmentations is not None:
            self.augmentations._set_seed()

        eval_set = eval_set if eval_set else []

        if loss_fn is None:
            self.loss_fn = self._default_loss
        else:
            self.loss_fn = loss_fn

        check_input(X_train)
        check_warm_start(warm_start, from_unsupervised)

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

        if scipy.sparse.issparse(X):
            dataloader = TBDataLoader(
                name="predict",
                dataset=SparsePredictDataset(X),
                batch_size=self.batch_size,
                predict=True,
            )
        else:
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

    def explain(self, X: np.ndarray, normalize: bool = False) -> Tuple[np.ndarray, Dict]:
        """Return local explanation.

        Parameters
        ----------
        X : tensor: `torch.Tensor` or matrix: `scipy.sparse.csr_matrix`
            Input data.
        normalize : bool (default False)
            Whether to normalize so that sum of features are equal to 1.

        Returns
        -------
        np.ndarray
            Importance per sample, per columns.
        dict
            Sparse matrix showing attention masks used by network.

        """
        self.network.eval()

        if scipy.sparse.issparse(X):
            dataloader = TBDataLoader(
                name="predict",
                dataset=SparsePredictDataset(X),
                batch_size=self.batch_size,
                predict=True,
            )
        else:
            dataloader = TBDataLoader(
                name="predict",
                dataset=PredictDataset(X),
                batch_size=self.batch_size,
                predict=True,
            )

        res_explain = []
        with torch.no_grad():
            for batch_nb, (data, _, _) in enumerate(dataloader):  # type: ignore
                data = data.to(self.device, non_blocking=True).float()  # type: ignore

                M_explain, masks = self.network.forward_masks(data)
                for key, value in masks.items():
                    masks[key] = csc_matrix.dot(value.cpu().detach().numpy(), self.reducing_matrix)
                original_feat_explain = csc_matrix.dot(M_explain.cpu().detach().numpy(), self.reducing_matrix)
                res_explain.append(original_feat_explain)

                if batch_nb == 0:
                    res_masks = masks
                else:
                    for key, value in masks.items():
                        res_masks[key] = np.vstack([res_masks[key], value])

        res_explain = np.vstack(res_explain)

        if normalize:
            res_explain /= np.sum(res_explain, axis=1)[:, None]

        return res_explain, res_masks

    def load_weights_from_unsupervised(self, unsupervised_model: "TabModel") -> None:
        """Load weights from a previously trained unsupervised TabNet model.

        Parameters
        ----------
        unsupervised_model : TabModel
            Previously trained unsupervised TabNet model.

        """
        update_state_dict = copy.deepcopy(self.network.state_dict())
        for param, weights in unsupervised_model.network.state_dict().items():
            if param.startswith("encoder"):
                new_param = "tabnet." + param
            else:
                new_param = param
            if self.network.state_dict().get(new_param) is not None:
                update_state_dict[new_param] = weights

        self.network.load_state_dict(update_state_dict)

    def load_class_attrs(self, class_attrs: Dict) -> None:
        """Load class attributes from a dictionary.

        Parameters
        ----------
        class_attrs : dict
            Dictionary of class attributes to set.

        """
        for attr_name, attr_value in class_attrs.items():
            setattr(self, attr_name, attr_value)

    def save_model(self, path: str) -> str:
        """Save TabNet model in two distinct files.

        Parameters
        ----------
        path : str
            Path of the model.

        Returns
        -------
        str
            Input filepath with ".zip" appended.

        """
        saved_params = {}
        init_params = {}
        for key, val in self.get_params().items():
            if isinstance(val, type):
                continue
            else:
                init_params[key] = val
        saved_params["init_params"] = init_params

        class_attrs = {"preds_mapper": self.preds_mapper}
        saved_params["class_attrs"] = class_attrs

        Path(path).mkdir(parents=True, exist_ok=True)

        with open(Path(path).joinpath("model_params.json"), "w", encoding="utf8") as f:
            json.dump(saved_params, f, cls=ComplexEncoder)

        torch.save(self.network.state_dict(), Path(path).joinpath("network.pt"))
        shutil.make_archive(path, "zip", path)
        shutil.rmtree(path)
        print(f"Successfully saved model at {path}.zip")
        return f"{path}.zip"

    def load_model(self, filepath: str) -> None:
        """Load TabNet model.

        Parameters
        ----------
        filepath : str
            Path of the model.

        Raises
        ------
        KeyError
            If the zip file is missing required components.

        """
        try:
            with zipfile.ZipFile(filepath) as z:
                with z.open("model_params.json") as f:
                    loaded_params = json.load(f)
                    loaded_params["init_params"]["device_name"] = self.device_name
                with z.open("network.pt") as f:
                    try:
                        saved_state_dict = torch.load(f, map_location=self.device)
                    except io.UnsupportedOperation:
                        saved_state_dict = torch.load(
                            io.BytesIO(f.read()),
                            map_location=self.device,
                        )
        except KeyError:
            raise KeyError("Your zip file is missing at least one component")

        self.__init__(**loaded_params["init_params"])  # type: ignore

        self._set_network()
        self.network.load_state_dict(saved_state_dict)
        self.network.eval()
        self.load_class_attrs(loaded_params["class_attrs"])

        return

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

        if self.augmentations is not None:
            X, y = self.augmentations(X, y)

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

        self.reducing_matrix = create_explain_matrix(
            self.network.input_dim,
            self.network.cat_emb_dim,
            self.network.cat_idxs,
            self.network.post_embed_dim,
        )

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
        raise NotImplementedError("users must define update_fit_params to use this base class")

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
        ...

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
        ...

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
        ...

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
        ...
