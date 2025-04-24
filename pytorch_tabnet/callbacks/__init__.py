"""Callback classes and utilities for TabNet training."""

import copy
import datetime
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch


class Callback:
    """Build new callbacks for TabNet training.

    This is an abstract base class for creating custom callbacks.
    """

    def __init__(self) -> None:
        """Initialize the Callback base class."""
        pass

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set parameters for the callback.

        Args:
            params (Dict[str, Any]): Parameters to set.

        """
        self.params = params

    def set_trainer(self, model: Any) -> None:
        """Set the trainer/model for the callback.

        Args:
            model (Any): The model or trainer instance.

        """
        self.trainer = model

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call at the beginning of an epoch.

        Args:
            epoch (int): Current epoch number.
            logs (Optional[Dict[str, Any]]): Additional logs.

        """
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call at the end of an epoch.

        Args:
            epoch (int): Current epoch number.
            logs (Optional[Dict[str, Any]]): Additional logs.

        """
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call at the beginning of a batch.

        Args:
            batch (int): Current batch number.
            logs (Optional[Dict[str, Any]]): Additional logs.

        """
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call at the end of a batch.

        Args:
            batch (int): Current batch number.
            logs (Optional[Dict[str, Any]]): Additional logs.

        """
        pass

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call at the beginning of training.

        Args:
            logs (Optional[Dict[str, Any]]): Additional logs.

        """
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call at the end of training.

        Args:
            logs (Optional[Dict[str, Any]]): Additional logs.

        """
        pass


@dataclass
class CallbackContainer:
    """Manage multiple callbacks during training."""

    callbacks: List[Callback] = field(default_factory=list)

    def append(self, callback: Callback) -> None:
        """Append a callback to the container.

        Args:
            callback (Callback): The callback to append.

        """
        self.callbacks.append(callback)

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set parameters for all callbacks in the container.

        Args:
            params (Dict[str, Any]): Parameters to set.

        """
        for callback in self.callbacks:
            callback.set_params(params)

    def set_trainer(self, trainer: Any) -> None:
        """Set the trainer for all callbacks in the container.

        Args:
            trainer (Any): The trainer or model instance.

        """
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_epoch_begin for all callbacks in the container.

        Args:
            epoch (int): Current epoch number.
            logs (Optional[Dict[str, Any]]): Additional logs.

        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_epoch_end for all callbacks in the container.

        Args:
            epoch (int): Current epoch number.
            logs (Optional[Dict[str, Any]]): Additional logs.

        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_batch_begin for all callbacks in the container.

        Args:
            batch (int): Current batch number.
            logs (Optional[Dict[str, Any]]): Additional logs.

        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_batch_end for all callbacks in the container.

        Args:
            batch (int): Current batch number.
            logs (Optional[Dict[str, Any]]): Additional logs.

        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_train_begin for all callbacks in the container.

        Args:
            logs (Optional[Dict[str, Any]]): Additional logs.

        """
        logs = logs or {}
        logs["start_time"] = time.time()
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_train_end for all callbacks in the container.

        Args:
            logs (Optional[Dict[str, Any]]): Additional logs.

        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)


@dataclass
class EarlyStopping(Callback):
    """Callback that stops training when a monitored metric has stopped improving."""

    early_stopping_metric: str
    is_maximize: bool
    tol: float = 0.0
    patience: int = 5

    def __post_init__(self) -> None:
        """Initialize EarlyStopping callback and set initial state."""
        self.best_epoch: int = 0
        self.stopped_epoch: int = 0
        self.wait: int = 0
        self.best_weights: Optional[Any] = None
        self.best_loss: float = np.inf
        if self.is_maximize:
            self.best_loss = -self.best_loss
        super().__init__()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Check for early stopping condition at the end of an epoch."""
        current_loss = logs.get(self.early_stopping_metric)
        if current_loss is None:
            return

        loss_change = current_loss - self.best_loss
        max_improved = self.is_maximize and loss_change > self.tol
        min_improved = (not self.is_maximize) and (-loss_change > self.tol)
        if max_improved or min_improved:
            self.best_loss = current_loss.item() if isinstance(current_loss, torch.Tensor) else current_loss
            self.best_epoch = epoch
            self.wait = 1
            self.best_weights = copy.deepcopy(self.trainer.network.state_dict())
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.trainer._stop_training = True
            self.wait += 1

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Restore best weights and print early stopping message at the end of training."""
        self.trainer.best_epoch = self.best_epoch
        self.trainer.best_cost = self.best_loss

        if self.best_weights is not None:
            self.trainer.network.load_state_dict(self.best_weights)

        if self.stopped_epoch > 0:
            msg = f"\nEarly stopping occurred at epoch {self.stopped_epoch}"
            msg += f" with best_epoch = {self.best_epoch} and " + f"best_{self.early_stopping_metric} = {round(self.best_loss, 5)}"
            print(msg)
        else:
            msg = (
                f"Stop training because you reached max_epochs = {self.trainer.max_epochs}"
                + f" with best_epoch = {self.best_epoch} and "
                + f"best_{self.early_stopping_metric} = {round(self.best_loss, 5)}"
            )
            print(msg)
        wrn_msg = "Best weights from best epoch are automatically used!"
        warnings.warn(wrn_msg, stacklevel=2)


@dataclass
class History(Callback):
    """Record events into a `History` object."""

    trainer: Any
    verbose: int = 1

    def __post_init__(self) -> None:
        """Initialize History callback and set counters."""
        super().__init__()
        self.samples_seen: float = 0.0
        self.total_time: float = 0.0

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Initialize history at the start of training.

        Args:
            logs (Optional[Dict[str, Any]]): Additional logs.

        """
        self.history: Dict[str, List[float]] = {"loss": []}
        self.history.update({"lr": []})
        self.history.update({name: [] for name in self.trainer._metrics_names})
        self.start_time: float = logs["start_time"]
        self.epoch_loss: float = 0.0

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Reset metrics at the start of an epoch.

        Args:
            epoch (int): Current epoch number.
            logs (Optional[Dict[str, Any]]): Additional logs.

        """
        self.epoch_metrics: Dict[str, float] = {"loss": 0.0}
        self.samples_seen = 0.0

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Update history and print metrics at the end of an epoch.

        Args:
            epoch (int): Current epoch number.
            logs (Optional[Dict[str, Any]]): Additional logs.

        """
        self.epoch_metrics["loss"] = self.epoch_loss
        for metric_name, metric_value in self.epoch_metrics.items():
            self.history[metric_name].append(metric_value)
        if self.verbose == 0:
            return
        if epoch % self.verbose != 0:
            return
        msg = f"epoch {epoch:<3}"
        for metric_name, metric_value in self.epoch_metrics.items():
            if metric_name != "lr":
                msg += f"| {metric_name:<3}: {np.round(metric_value, 5):<8}"
        self.total_time = int(time.time() - self.start_time)
        msg += f"|  {str(datetime.timedelta(seconds=self.total_time)) + 's':<6}"
        print(msg)

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Update epoch loss after a batch.

        Args:
            batch (int): Current batch number.
            logs (Optional[Dict[str, Any]]): Additional logs. Must include 'batch_size' and 'loss'.

        """
        batch_size: int = logs["batch_size"]
        self.epoch_loss = (self.samples_seen * self.epoch_loss + batch_size * logs["loss"]) / (self.samples_seen + batch_size)
        self.samples_seen += batch_size

    def __getitem__(self, name: str) -> List[float]:
        """Return metric history by name.

        Args:
            name (str): Name of the metric.

        Returns:
            List[float]: List of metric values.

        """
        return self.history[name]

    def __repr__(self) -> str:
        """Return string representation of the history object."""
        return str(self.history)

    def __str__(self) -> str:
        """Return string representation of the history object."""
        return str(self.history)


@dataclass
class LRSchedulerCallback(Callback):
    """Callback that updates the learning rate according to a scheduler."""

    scheduler_fn: Any
    optimizer: Any
    scheduler_params: Dict[str, Any]
    early_stopping_metric: str
    is_batch_level: bool = False

    def __post_init__(
        self,
    ) -> None:
        """Initialize the learning rate scheduler callback."""
        self.is_metric_related: bool = hasattr(self.scheduler_fn, "is_better")
        self.scheduler: Any = self.scheduler_fn(self.optimizer, **self.scheduler_params)
        super().__init__()

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Update the learning rate at the end of a batch if batch-level scheduling is enabled."""
        if self.is_batch_level:
            self.scheduler.step()
        else:
            pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Update the learning rate at the end of an epoch if epoch-level scheduling is enabled."""
        current_loss = logs.get(self.early_stopping_metric)
        if current_loss is None:
            return
        if self.is_batch_level:
            pass
        else:
            if self.is_metric_related:
                self.scheduler.step(current_loss)
            else:
                self.scheduler.step()
