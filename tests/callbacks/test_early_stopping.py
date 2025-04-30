from unittest.mock import MagicMock, patch

import numpy as np
import torch

from pytorch_tabnet.callbacks import EarlyStopping


class TestEarlyStopping:
    """Test the EarlyStopping callback functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.metric = "val_loss"
        self.mock_trainer = MagicMock()
        self.mock_trainer._stop_training = False  # Initialize the attribute explicitly
        self.mock_trainer.network = MagicMock()
        self.mock_trainer.network.state_dict.return_value = {"layer1": torch.ones(1)}
        self.mock_state_dict = {"layer1": torch.ones(1)}

    def test_init_minimize(self):
        """Test initialization for minimizing metric."""
        es = EarlyStopping(early_stopping_metric=self.metric, is_maximize=False)
        assert es.early_stopping_metric == self.metric
        assert es.is_maximize is False
        assert es.tol == 0.0
        assert es.patience == 5
        assert es.best_loss == np.inf
        assert es.best_epoch == 0
        assert es.stopped_epoch == 0
        assert es.wait == 0
        assert es.best_weights is None

    def test_init_maximize(self):
        """Test initialization for maximizing metric."""
        es = EarlyStopping(early_stopping_metric=self.metric, is_maximize=True)
        assert es.is_maximize is True
        assert es.best_loss == -np.inf

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        es = EarlyStopping(early_stopping_metric=self.metric, is_maximize=False, tol=0.01, patience=10)
        assert es.tol == 0.01
        assert es.patience == 10

    def test_on_epoch_end_missing_metric(self):
        """Test on_epoch_end when metric is missing from logs."""
        es = EarlyStopping(early_stopping_metric=self.metric, is_maximize=False)
        es.set_trainer(self.mock_trainer)

        # Should not change any state when metric is missing
        logs = {"other_metric": 0.5}
        es.on_epoch_end(0, logs)

        assert es.best_loss == np.inf
        assert es.best_epoch == 0
        assert es.wait == 0
        assert es.best_weights is None

    def test_on_epoch_end_improve_minimize(self):
        """Test on_epoch_end when metric improves for minimization."""
        es = EarlyStopping(early_stopping_metric=self.metric, is_maximize=False)
        es.set_trainer(self.mock_trainer)

        # First epoch - should set best loss
        logs = {self.metric: 0.5}
        es.on_epoch_end(0, logs)

        assert es.best_loss == 0.5
        assert es.best_epoch == 0
        assert es.wait == 1
        assert es.best_weights is not None

        # Improvement - should update best loss
        logs = {self.metric: 0.3}
        es.on_epoch_end(1, logs)

        assert es.best_loss == 0.3
        assert es.best_epoch == 1
        assert es.wait == 1

        # Improvement smaller than tolerance
        es.tol = 0.2
        logs = {self.metric: 0.2}
        es.on_epoch_end(2, logs)

        assert es.best_loss == 0.3
        assert es.best_epoch == 1
        assert es.wait == 2

    def test_on_epoch_end_improve_maximize(self):
        """Test on_epoch_end when metric improves for maximization."""
        es = EarlyStopping(early_stopping_metric=self.metric, is_maximize=True)
        es.set_trainer(self.mock_trainer)

        # First epoch - should set best loss
        logs = {self.metric: 0.5}
        es.on_epoch_end(0, logs)

        assert es.best_loss == 0.5
        assert es.best_epoch == 0
        assert es.wait == 1
        assert es.best_weights is not None

        # Improvement - should update best loss
        logs = {self.metric: 0.7}
        es.on_epoch_end(1, logs)

        assert es.best_loss == 0.7
        assert es.best_epoch == 1
        assert es.wait == 1

        # Improvement smaller than tolerance
        es.tol = 0.2
        logs = {self.metric: 0.8}
        es.on_epoch_end(2, logs)

        assert es.best_loss == 0.7
        assert es.best_epoch == 1
        assert es.wait == 2

    def test_on_epoch_end_tensor_value(self):
        """Test on_epoch_end with torch.Tensor value."""
        es = EarlyStopping(early_stopping_metric=self.metric, is_maximize=False)
        es.set_trainer(self.mock_trainer)

        logs = {self.metric: torch.tensor(0.5)}
        es.on_epoch_end(0, logs)

        assert es.best_loss == 0.5
        assert es.best_epoch == 0
        assert es.wait == 1
        assert es.best_weights is not None

    def test_on_epoch_end_early_stop(self):
        """Test on_epoch_end triggering early stopping."""
        es = EarlyStopping(early_stopping_metric=self.metric, is_maximize=False, patience=2)
        es.set_trainer(self.mock_trainer)

        # First epoch
        logs = {self.metric: 0.5}
        es.on_epoch_end(0, logs)

        assert es.wait == 1
        assert not self.mock_trainer._stop_training

        # No improvement - wait=2
        logs = {self.metric: 0.6}
        es.on_epoch_end(1, logs)

        assert es.wait == 2
        assert not self.mock_trainer._stop_training

        # No improvement - patience exceeded, should stop
        logs = {self.metric: 0.7}
        es.on_epoch_end(2, logs)

        assert es.wait == 3
        assert self.mock_trainer._stop_training
        assert es.stopped_epoch == 2

    @patch("warnings.warn")
    def test_on_train_end_early_stopped(self, mock_warn):
        """Test on_train_end when early stopping occurred."""
        es = EarlyStopping(early_stopping_metric=self.metric, is_maximize=False)
        es.set_trainer(self.mock_trainer)

        # Set state as if early stopping occurred
        es.best_epoch = 5
        es.best_loss = 0.3
        es.stopped_epoch = 10
        es.best_weights = self.mock_state_dict

        with patch("builtins.print") as mock_print:
            es.on_train_end()

            # Check that the best weights were loaded
            self.mock_trainer.network.load_state_dict.assert_called_once_with(self.mock_state_dict)

            # Check that the best metrics were set on the trainer
            assert self.mock_trainer.best_epoch == 5
            assert self.mock_trainer.best_cost == 0.3

            # Check that the appropriate message was printed
            mock_print.assert_called_once()
            call_args = mock_print.call_args[0][0]
            assert f"Early stopping occurred at epoch {es.stopped_epoch}" in call_args
            assert f"best_epoch = {es.best_epoch}" in call_args
            assert f"best_{es.early_stopping_metric} = {round(es.best_loss, 5)}" in call_args

            # Check that the warning was issued
            mock_warn.assert_called_once_with("Best weights from best epoch are automatically used!", stacklevel=2)

    @patch("warnings.warn")
    def test_on_train_end_max_epochs(self, mock_warn):
        """Test on_train_end when max epochs was reached."""
        es = EarlyStopping(early_stopping_metric=self.metric, is_maximize=False)
        es.set_trainer(self.mock_trainer)
        es.trainer.max_epochs = 100

        # Set state as if max epochs was reached (stopped_epoch = 0)
        es.best_epoch = 95
        es.best_loss = 0.3
        es.stopped_epoch = 0
        es.best_weights = self.mock_state_dict

        with patch("builtins.print") as mock_print:
            es.on_train_end()

            # Check that the best weights were loaded
            self.mock_trainer.network.load_state_dict.assert_called_once_with(self.mock_state_dict)

            # Check that the best metrics were set on the trainer
            assert self.mock_trainer.best_epoch == 95
            assert self.mock_trainer.best_cost == 0.3

            # Check that the appropriate message was printed
            mock_print.assert_called_once()
            call_args = mock_print.call_args[0][0]
            assert f"Stop training because you reached max_epochs = {es.trainer.max_epochs}" in call_args
            assert f"best_epoch = {es.best_epoch}" in call_args
            assert f"best_{es.early_stopping_metric} = {round(es.best_loss, 5)}" in call_args

            # Check that the warning was issued
            mock_warn.assert_called_once_with("Best weights from best epoch are automatically used!", stacklevel=2)
