import typing  # Added for Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.optim as optim

from pytorch_tabnet.callbacks import Callback, LRSchedulerCallback
from pytorch_tabnet.tab_models.abstract_models import TabSupervisedModel


@dataclass
class MockTabModel(TabSupervisedModel):
    """Mock TabModel for testing."""

    _default_loss: torch.nn.Module = None  # Reverted to torch.nn.Module
    _default_metric: str = None
    updated_weights: bool = False
    preds_mapper: typing.Callable[[np.ndarray], np.ndarray] = None

    def update_fit_params(self, X_train, y_train, eval_set, weights) -> None:  # Added return type
        self.output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1
        self._default_loss = torch.nn.MSELoss()
        self._default_metric = "mse"
        self.preds_mapper = lambda x: x

    def compute_loss(self, y_score: torch.Tensor, y_true: torch.Tensor, weights: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
        if weights is None:
            return self._default_loss(y_score, y_true)
        # For simplicity, ignoring weights or assuming it's handled by a wrapper
        return self._default_loss(y_score, y_true)

    def prepare_target(self, y: np.ndarray) -> np.ndarray:  # Reverted to np.ndarray
        return y  # Keep as numpy array as expected by check_data_general

    def predict_func(self, res: np.ndarray) -> np.ndarray:  # Added type hints
        return res


# Start writing tests using pytest
def test_fit_predict_simple() -> None:  # Added return type
    """Test fitting and predicting with simple data."""
    X_train = np.array([[0, 1], [1, 0]])
    y_train = np.array([[1], [0]])
    model = MockTabModel()
    model.fit(X_train, y_train, max_epochs=1)  # Call fit before optimizer check
    assert model._optimizer is not None
    assert isinstance(model._optimizer, optim.Adam)
    predictions = model.predict(X_train)
    assert predictions.shape == y_train.shape


def test_fit_predict_more_complex() -> None:  # Added return type
    """Test with slightly more complex data and parameters."""
    X_train = np.random.rand(10, 3)
    y_train = np.random.rand(10, 2)  # Multi-output
    model = MockTabModel(n_d=4, n_a=4, n_steps=2)
    model.fit(X_train, y_train, max_epochs=1)
    predictions = model.predict(X_train)
    assert predictions.shape == y_train.shape


# New test for custom callbacks and scheduler
def test_custom_callbacks_and_scheduler() -> None:  # Added return type
    """Test fit with custom callbacks and a learning rate scheduler."""
    X_train = np.array([[0, 1], [1, 0]])
    y_train = np.array([[1], [0]])

    class MyCustomCallback(Callback):
        def __init__(self) -> None:  # Added return type
            super().__init__()
            self.on_epoch_begin_called = False

        def on_epoch_begin(self, epoch: int, logs: typing.Optional[dict] = None) -> None:  # Added type hints
            self.on_epoch_begin_called = True

    custom_callback = MyCustomCallback()

    def simple_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:  # Changed to LRScheduler
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    model = MockTabModel(scheduler_fn=simple_scheduler, scheduler_params={})

    model.fit(X_train, y_train, max_epochs=1, callbacks=[custom_callback])

    assert custom_callback.on_epoch_begin_called, "Custom callback was not triggered"
    # Check for LRSchedulerCallback and its scheduler attribute
    lr_scheduler_callback = None
    for cb in model._callback_container.callbacks:
        if isinstance(cb, LRSchedulerCallback):
            lr_scheduler_callback = cb
            break
    assert lr_scheduler_callback is not None, "LRSchedulerCallback not found in callback container"
    assert lr_scheduler_callback.scheduler is not None, "Scheduler was not set on LRSchedulerCallback"
    assert any(isinstance(cb, torch.optim.lr_scheduler.LambdaLR) for cb in model._callback_container.callbacks) or any(
        isinstance(cb, LRSchedulerCallback) for cb in model._callback_container.callbacks
    ), "LRSchedulerCallback not found in callback container"


def test_explain() -> None:  # Added return type
    """Test the explain method."""
    X_train = np.random.rand(5, 2)
    model = MockTabModel()
    model.fit(X_train, np.random.rand(5, 1), max_epochs=1)
    explain_matrix, _masks = model.explain(X_train)
    assert explain_matrix.shape == X_train.shape
