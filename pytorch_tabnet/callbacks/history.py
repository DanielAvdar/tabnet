import datetime
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from pytorch_tabnet.callbacks import Callback


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
