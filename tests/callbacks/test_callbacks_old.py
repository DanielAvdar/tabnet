import time

import pytest

from pytorch_tabnet.callbacks import (
    Callback,
    CallbackContainer,
    EarlyStopping,
    History,
    LRSchedulerCallback,
)


# Define a sample trainer class for testing purposes
class Trainer:
    def __init__(self):
        self.network = MockNetwork()  # Use a mock network for testing
        self.max_epochs = 5
        self._stop_training = False
        self._metrics_names = ["loss"]

    def load_state_dict(self):
        return


class MockNetwork:
    def __init__(self):
        self.weights = 1

    def state_dict(self):
        return {"weights": self.weights}

    def load_state_dict(self, weights):
        self.weights = weights["weights"]


@pytest.fixture
def trainer():
    return Trainer()


@pytest.fixture
def logs():
    return {"batch_size": 32, "loss": 0.5, "start_time": time.time()}


def test_callback():
    callback = Callback()
    assert isinstance(callback, Callback)

    # Test setting params
    params = {"param1": "value1", "param2": "value2"}
    callback.set_params(params)
    assert callback.params == params

    # Test all methods for base coverage
    callback.on_epoch_begin(0, None)
    callback.on_epoch_end(0, None)
    callback.on_batch_begin(0, None)
    callback.on_batch_end(0, None)
    callback.on_train_begin(None)
    callback.on_train_end(None)


def test_callback_container(trainer):
    callback1 = Callback()
    callback2 = Callback()
    container = CallbackContainer([callback1, callback2])
    container.set_trainer(trainer)
    assert container.trainer == trainer
    assert callback1.trainer == trainer
    assert callback2.trainer == trainer

    # Test append method
    callback3 = Callback()
    container.append(callback3)
    assert len(container.callbacks) == 3
    assert callback3 in container.callbacks

    # Test set_params method
    params = {"param1": "value1", "param2": "value2"}
    container.set_params(params)
    for callback in container.callbacks:
        assert callback.params == params

    # Test empty logs handling in all methods
    container.on_train_begin()
    container.on_epoch_begin(0)
    container.on_batch_begin(0)
    container.on_batch_end(0)
    container.on_epoch_end(0)
    container.on_train_end()


def test_early_stopping_minimize(trainer, logs):
    early_stopping = EarlyStopping(early_stopping_metric="loss", is_maximize=False, patience=2, tol=0.01)
    early_stopping.set_trainer(trainer)

    # Simulate training for a few epochs
    for epoch in range(5):
        logs["loss"] = 0.5 - epoch * 0.1  # Monotonically decreasing Loss
        early_stopping.on_epoch_end(epoch, logs)

        if trainer._stop_training:
            break

    # loss improves each time, so training shouldn't stop early
    assert trainer._stop_training is False
    assert abs(early_stopping.best_loss - 0.1) < 0.01
    assert early_stopping.best_epoch == 4


def test_early_stopping_maximize(trainer, logs):
    early_stopping = EarlyStopping(early_stopping_metric="loss", is_maximize=True, patience=2, tol=0.1)
    early_stopping.set_trainer(trainer)

    # Simulate training for a few epochs
    for epoch in range(5):
        logs["loss"] = 0.5 + epoch * 0.1  # Monotonically increasing Loss
        early_stopping.on_epoch_end(epoch, logs)

        if trainer._stop_training:
            break

    # loss improves each time, so training shouldn't stop early
    assert trainer._stop_training is False
    # assert early_stopping.best_loss == 0.9
    assert abs(early_stopping.best_loss - 0.8) < 0.01
    assert early_stopping.best_epoch == 3


def test_early_stopping_patience(trainer, logs):
    early_stopping = EarlyStopping(early_stopping_metric="loss", is_maximize=False, patience=2, tol=0.1)
    early_stopping.set_trainer(trainer)

    losses = [0.5, 0.4, 0.5, 0.6, 0.7]

    for epoch, loss in enumerate(losses):
        logs["loss"] = loss
        early_stopping.on_epoch_end(epoch, logs)
        if trainer._stop_training:
            break

    assert epoch == 2  # Training should stop early at epoch 3 (index from 0)
    assert (early_stopping.best_loss - 0.5) < 0.01
    assert early_stopping.best_epoch == 0


def test_early_stopping_on_train_end(trainer, logs):
    """Test the on_train_end method of EarlyStopping with stopped_epoch > 0."""
    early_stopping = EarlyStopping(early_stopping_metric="loss", is_maximize=False, patience=2, tol=0.1)
    early_stopping.set_trainer(trainer)
    early_stopping.best_epoch = 3
    early_stopping.best_loss = 0.3
    early_stopping.stopped_epoch = 5
    early_stopping.best_weights = {"weights": 0.5}

    # Test with stopped_epoch > 0
    early_stopping.on_train_end(logs)
    assert trainer.best_epoch == 3
    assert trainer.best_cost == 0.3
    assert trainer.network.weights == 0.5

    # Test with stopped_epoch = 0 (reaching max_epochs)
    early_stopping = EarlyStopping(early_stopping_metric="loss", is_maximize=False, patience=2, tol=0.1)
    early_stopping.set_trainer(trainer)
    early_stopping.best_epoch = 4
    early_stopping.best_loss = 0.2
    early_stopping.stopped_epoch = 0
    early_stopping.best_weights = {"weights": 0.7}

    early_stopping.on_train_end(logs)
    assert trainer.best_epoch == 4
    assert trainer.best_cost == 0.2
    assert trainer.network.weights == 0.7


def test_early_stopping_missing_metric(trainer, logs):
    """Test EarlyStopping behavior when the monitored metric is missing."""
    early_stopping = EarlyStopping(early_stopping_metric="accuracy", is_maximize=True, patience=2)
    early_stopping.set_trainer(trainer)

    # No 'accuracy' metric in logs, should not change the state
    initial_loss = early_stopping.best_loss
    early_stopping.on_epoch_end(0, logs)
    assert early_stopping.best_loss == initial_loss
    assert early_stopping.wait == 0


def test_history(trainer, logs):
    history = History(trainer)
    history.on_train_begin(logs)
    for epoch in range(2):
        history.on_epoch_begin(epoch, logs)
        for batch in range(2):
            history.on_batch_end(batch, logs)
        history.on_epoch_end(epoch, logs)
        # Check if values are appended to history after each epoch
        assert len(history.history["loss"]) == epoch + 1


def test_history_getitem_repr_str(trainer, logs):
    """Test the __getitem__, __repr__, and __str__ methods of History."""
    history = History(trainer)
    history.on_train_begin(logs)

    # Add some data
    history.on_epoch_begin(0, logs)
    history.on_batch_end(0, logs)
    history.on_epoch_end(0, logs)

    # Test __getitem__
    assert history["loss"] == history.history["loss"]

    # Test __repr__ and __str__
    assert repr(history) == str(history.history)
    assert str(history) == str(history.history)


def test_history_verbose_settings(trainer, logs):
    """Test History with different verbose settings."""
    # Test with verbose=0 (no output)
    history = History(trainer, verbose=0)
    history.on_train_begin(logs)
    history.on_epoch_begin(0, logs)
    history.on_batch_end(0, logs)
    history.on_epoch_end(0, logs)  # Should not print

    # Test with verbose=2 (print every 2 epochs)
    history = History(trainer, verbose=2)
    history.on_train_begin(logs)

    # First epoch (should not print)
    history.on_epoch_begin(1, logs)
    history.on_batch_end(0, logs)
    history.on_epoch_end(1, logs)

    # Second epoch (should print)
    history.on_epoch_begin(2, logs)
    history.on_batch_end(0, logs)
    history.on_epoch_end(2, logs)


class MockScheduler:
    def __init__(self, optimizer, **params):
        self.optimizer = optimizer
        self.params = params
        self.last_lr = None
        self.call_count = 0
        self.is_better = True  # Only for testing

    def step(self, loss=None):
        self.call_count += 1
        for param_group in self.optimizer.param_groups:
            self.last_lr = param_group["lr"] * 2
            param_group["lr"] = self.last_lr


class MockOptimizer:
    def __init__(self):
        self.param_groups = [{"lr": 0.1}]


def test_lr_scheduler(trainer, logs):
    optimizer = MockOptimizer()
    lr_scheduler = LRSchedulerCallback(MockScheduler, optimizer, {"param1": 1}, "loss")
    lr_scheduler.set_trainer(trainer)

    initial_lr = optimizer.param_groups[0]["lr"]

    for epoch in range(2):
        lr_scheduler.on_epoch_end(epoch, logs)
        # Check if LR is updated after each epoch
        assert optimizer.param_groups[0]["lr"] > initial_lr
        initial_lr = optimizer.param_groups[0]["lr"]

    assert lr_scheduler.scheduler.call_count == 2


def test_lr_scheduler_batch_level(trainer, logs):
    """Test LRSchedulerCallback with batch-level scheduling."""
    optimizer = MockOptimizer()
    lr_scheduler = LRSchedulerCallback(MockScheduler, optimizer, {"param1": 1}, "loss", is_batch_level=True)
    lr_scheduler.set_trainer(trainer)

    initial_lr = optimizer.param_groups[0]["lr"]

    # First, test batch-level updates
    for batch in range(2):
        lr_scheduler.on_batch_end(batch, logs)
        # Check if LR is updated after each batch
        assert optimizer.param_groups[0]["lr"] > initial_lr
        initial_lr = optimizer.param_groups[0]["lr"]

    # Then, test that epoch-level updates don't happen when is_batch_level=True
    initial_lr = optimizer.param_groups[0]["lr"]
    call_count = lr_scheduler.scheduler.call_count

    lr_scheduler.on_epoch_end(0, logs)
    # Should not have updated in on_epoch_end
    assert lr_scheduler.scheduler.call_count == call_count


def test_lr_scheduler_missing_metric(trainer, logs):
    """Test LRSchedulerCallback behavior when the monitored metric is missing."""
    optimizer = MockOptimizer()
    lr_scheduler = LRSchedulerCallback(
        MockScheduler,
        optimizer,
        {"param1": 1},
        "accuracy",  # Metric not in logs
    )
    lr_scheduler.set_trainer(trainer)

    initial_lr = optimizer.param_groups[0]["lr"]
    call_count = lr_scheduler.scheduler.call_count

    # Should not update when metric is missing
    lr_scheduler.on_epoch_end(0, {})
    assert lr_scheduler.scheduler.call_count == call_count
    assert optimizer.param_groups[0]["lr"] == initial_lr
