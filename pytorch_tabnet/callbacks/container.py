import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pytorch_tabnet.callbacks import Callback


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
