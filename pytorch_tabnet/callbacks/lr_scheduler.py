from dataclasses import dataclass
from typing import Any, Dict, Optional

from pytorch_tabnet.callbacks import Callback


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
