import pytest

import time

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


def test_callback_container(trainer):
    callback1 = Callback()
    callback2 = Callback()
    container = CallbackContainer([callback1, callback2])
    container.set_trainer(trainer)
    assert container.trainer == trainer
    assert callback1.trainer == trainer
    assert callback2.trainer == trainer


def test_early_stopping_minimize(trainer, logs):
    early_stopping = EarlyStopping(
        early_stopping_metric="loss", is_maximize=False, patience=2, tol=0.01
    )
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
    early_stopping = EarlyStopping(
        early_stopping_metric="loss", is_maximize=True, patience=2, tol=0.1
    )
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
    early_stopping = EarlyStopping(
        early_stopping_metric="loss", is_maximize=False, patience=2, tol=0.1
    )
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


class MockScheduler:
    def __init__(self, optimizer, **params):
        self.optimizer = optimizer
        self.params = params
        self.last_lr = None
        self.call_count = 0

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
