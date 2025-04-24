import copy
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from pytorch_tabnet.callbacks import Callback


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
