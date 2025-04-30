import time
from unittest.mock import MagicMock, patch

from pytorch_tabnet.callbacks import History


class TestHistory:
    """Test the History callback functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_trainer = MagicMock()
        self.mock_trainer._metrics_names = ["accuracy", "precision", "recall"]
        self.verbose = 1
        self.history = History(trainer=self.mock_trainer, verbose=self.verbose)
        self.test_logs = {"batch_size": 32, "loss": 0.5, "start_time": 12345}

    def test_init(self):
        """Test initialization of History."""
        history = History(trainer=self.mock_trainer)
        assert history.trainer == self.mock_trainer
        assert history.verbose == 1
        assert history.samples_seen == 0.0
        assert history.total_time == 0.0

        history = History(trainer=self.mock_trainer, verbose=2)
        assert history.verbose == 2

    def test_on_train_begin(self):
        """Test on_train_begin method."""
        self.history.on_train_begin(self.test_logs)

        # Check that history contains required keys
        assert "loss" in self.history.history
        assert "lr" in self.history.history

        # Check that metrics from trainer were added
        for metric in self.mock_trainer._metrics_names:
            assert metric in self.history.history

        # Check that start time was set
        assert hasattr(self.history, "start_time")
        assert self.history.start_time == 12345

        # Check that epoch_loss was initialized
        assert hasattr(self.history, "epoch_loss")
        assert self.history.epoch_loss == 0.0

    def test_on_epoch_begin(self):
        """Test on_epoch_begin method."""
        self.history.on_epoch_begin(0)

        # Check that epoch_metrics was initialized with loss
        assert hasattr(self.history, "epoch_metrics")
        assert "loss" in self.history.epoch_metrics
        assert self.history.epoch_metrics["loss"] == 0.0

        # Check that samples_seen was reset
        assert self.history.samples_seen == 0.0

    def test_on_batch_end(self):
        """Test on_batch_end method."""
        # Initialize history
        self.history.epoch_loss = 0.0
        self.history.samples_seen = 0.0

        # First batch
        batch_logs = {"batch_size": 32, "loss": 0.5}
        self.history.on_batch_end(0, batch_logs)

        assert self.history.epoch_loss == 0.5
        assert self.history.samples_seen == 32

        # Second batch with different loss
        batch_logs = {"batch_size": 32, "loss": 0.3}
        self.history.on_batch_end(1, batch_logs)

        # Check that loss is weighted average
        expected_loss = (32 * 0.5 + 32 * 0.3) / 64
        assert abs(self.history.epoch_loss - expected_loss) < 1e-6
        assert self.history.samples_seen == 64

    def test_on_epoch_end_with_verbose(self):
        """Test on_epoch_end method with verbose output."""
        # Set up history state
        self.history.on_train_begin({"start_time": time.time() - 100})
        self.history.on_epoch_begin(0)
        self.history.epoch_loss = 0.5
        self.history.epoch_metrics = {"loss": 0.5, "accuracy": 0.8}

        with patch("builtins.print") as mock_print:
            self.history.on_epoch_end(0)

            # Check that history was updated
            assert self.history.history["loss"] == [0.5]

            # Check that print was called
            mock_print.assert_called_once()
            # Check that the printed message contains the metrics
            call_args = mock_print.call_args[0][0]
            assert "epoch 0" in call_args
            assert "loss: 0.5" in call_args
            assert "accuracy: 0.8" in call_args

    def test_on_epoch_end_no_verbose(self):
        """Test on_epoch_end method without verbose output."""
        # Set up history with verbose=0
        self.history = History(trainer=self.mock_trainer, verbose=0)
        self.history.on_train_begin({"start_time": time.time()})
        self.history.on_epoch_begin(0)
        self.history.epoch_loss = 0.5
        self.history.epoch_metrics = {"loss": 0.5, "accuracy": 0.8}

        with patch("builtins.print") as mock_print:
            self.history.on_epoch_end(0)

            # Check that history was updated
            assert self.history.history["loss"] == [0.5]

            # Check that print was not called
            mock_print.assert_not_called()

    def test_on_epoch_end_verbose_interval(self):
        """Test on_epoch_end with verbose interval."""
        # Set up history with verbose=2
        self.history = History(trainer=self.mock_trainer, verbose=2)
        self.history.on_train_begin({"start_time": time.time()})

        with patch("builtins.print") as mock_print:
            # Epoch 0 (multiple of 2, should print)
            self.history.on_epoch_begin(0)
            self.history.epoch_loss = 0.5
            self.history.epoch_metrics = {"loss": 0.5}
            self.history.on_epoch_end(0)
            assert mock_print.call_count == 1

            # Epoch 1 (not multiple of 2, should not print)
            mock_print.reset_mock()
            self.history.on_epoch_begin(1)
            self.history.epoch_loss = 0.4
            self.history.epoch_metrics = {"loss": 0.4}
            self.history.on_epoch_end(1)
            assert mock_print.call_count == 0

            # Epoch 2 (multiple of 2, should print)
            mock_print.reset_mock()
            self.history.on_epoch_begin(2)
            self.history.epoch_loss = 0.3
            self.history.epoch_metrics = {"loss": 0.3}
            self.history.on_epoch_end(2)
            assert mock_print.call_count == 1

    def test_getitem(self):
        """Test __getitem__ method."""
        self.history.history = {"loss": [0.5, 0.4, 0.3], "accuracy": [0.7, 0.8, 0.9]}

        assert self.history["loss"] == [0.5, 0.4, 0.3]
        assert self.history["accuracy"] == [0.7, 0.8, 0.9]

    def test_repr_and_str(self):
        """Test __repr__ and __str__ methods."""
        self.history.history = {"loss": [0.5, 0.4], "accuracy": [0.7, 0.8]}

        assert repr(self.history) == str(self.history.history)
        assert str(self.history) == str(self.history.history)
