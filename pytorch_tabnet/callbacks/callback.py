from typing import Any, Dict, Optional


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
