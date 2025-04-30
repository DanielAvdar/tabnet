from unittest.mock import MagicMock

from pytorch_tabnet.callbacks import LRSchedulerCallback


class TestLRSchedulerCallback:
    """Test the LRSchedulerCallback class functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.metric = "val_loss"
        self.mock_optimizer = MagicMock()
        self.mock_scheduler = MagicMock()

        # Mock scheduler factory function
        self.mock_scheduler_fn = MagicMock()
        self.mock_scheduler_fn.return_value = self.mock_scheduler

        self.scheduler_params = {"factor": 0.1, "patience": 10}
        self.test_logs = {self.metric: 0.5}

    def test_init_epoch_level(self):
        """Test initialization for epoch-level scheduler."""
        # Standard scheduler without is_better attribute
        # Remove is_better attribute if it exists from previous tests
        if hasattr(self.mock_scheduler_fn, "is_better"):
            delattr(self.mock_scheduler_fn, "is_better")

        callback = LRSchedulerCallback(
            scheduler_fn=self.mock_scheduler_fn,
            optimizer=self.mock_optimizer,
            scheduler_params=self.scheduler_params,
            early_stopping_metric=self.metric,
            is_batch_level=False,
        )

        assert callback.scheduler_fn == self.mock_scheduler_fn
        assert callback.optimizer == self.mock_optimizer
        assert callback.scheduler_params == self.scheduler_params
        assert callback.early_stopping_metric == self.metric
        assert callback.is_batch_level is False
        assert callback.is_metric_related is False
        assert callback.scheduler == self.mock_scheduler

        # Verify scheduler was created with correct parameters
        self.mock_scheduler_fn.assert_called_once_with(self.mock_optimizer, **self.scheduler_params)

    def test_init_batch_level(self):
        """Test initialization for batch-level scheduler."""
        # Remove is_better attribute if it exists from previous tests
        if hasattr(self.mock_scheduler_fn, "is_better"):
            delattr(self.mock_scheduler_fn, "is_better")

        callback = LRSchedulerCallback(
            scheduler_fn=self.mock_scheduler_fn,
            optimizer=self.mock_optimizer,
            scheduler_params=self.scheduler_params,
            early_stopping_metric=self.metric,
            is_batch_level=True,
        )

        assert callback.is_batch_level is True

    def test_init_metric_related(self):
        """Test initialization for metric-related scheduler."""
        # Add is_better attribute to mock scheduler_fn to simulate ReduceLROnPlateau
        self.mock_scheduler_fn.is_better = True

        callback = LRSchedulerCallback(
            scheduler_fn=self.mock_scheduler_fn,
            optimizer=self.mock_optimizer,
            scheduler_params=self.scheduler_params,
            early_stopping_metric=self.metric,
        )

        assert callback.is_metric_related is True

    def test_on_batch_end_batch_level(self):
        """Test on_batch_end for batch-level scheduler."""
        # Remove is_better attribute if it exists from previous tests
        if hasattr(self.mock_scheduler_fn, "is_better"):
            delattr(self.mock_scheduler_fn, "is_better")

        callback = LRSchedulerCallback(
            scheduler_fn=self.mock_scheduler_fn,
            optimizer=self.mock_optimizer,
            scheduler_params=self.scheduler_params,
            early_stopping_metric=self.metric,
            is_batch_level=True,
        )

        # Reset any previous calls
        self.mock_scheduler.step.reset_mock()

        callback.on_batch_end(0, self.test_logs)

        # Verify scheduler.step() was called
        self.mock_scheduler.step.assert_called_once()

    def test_on_batch_end_epoch_level(self):
        """Test on_batch_end for epoch-level scheduler (should do nothing)."""
        # Remove is_better attribute if it exists from previous tests
        if hasattr(self.mock_scheduler_fn, "is_better"):
            delattr(self.mock_scheduler_fn, "is_better")

        callback = LRSchedulerCallback(
            scheduler_fn=self.mock_scheduler_fn,
            optimizer=self.mock_optimizer,
            scheduler_params=self.scheduler_params,
            early_stopping_metric=self.metric,
            is_batch_level=False,
        )

        # Reset any previous calls
        self.mock_scheduler.step.reset_mock()

        callback.on_batch_end(0, self.test_logs)

        # Verify scheduler.step() was not called
        self.mock_scheduler.step.assert_not_called()

    def test_on_epoch_end_epoch_level_standard(self):
        """Test on_epoch_end for standard epoch-level scheduler."""
        # Remove is_better attribute if it exists from previous tests
        if hasattr(self.mock_scheduler_fn, "is_better"):
            delattr(self.mock_scheduler_fn, "is_better")

        callback = LRSchedulerCallback(
            scheduler_fn=self.mock_scheduler_fn,
            optimizer=self.mock_optimizer,
            scheduler_params=self.scheduler_params,
            early_stopping_metric=self.metric,
            is_batch_level=False,
        )

        # Reset any previous calls
        self.mock_scheduler.step.reset_mock()

        callback.on_epoch_end(0, self.test_logs)

        # Since is_metric_related is False, step() should be called without arguments
        self.mock_scheduler.step.assert_called_once()

    def test_on_epoch_end_epoch_level_metric_related(self):
        """Test on_epoch_end for metric-related epoch-level scheduler."""
        # Add is_better attribute to simulate ReduceLROnPlateau
        self.mock_scheduler_fn.is_better = True

        callback = LRSchedulerCallback(
            scheduler_fn=self.mock_scheduler_fn,
            optimizer=self.mock_optimizer,
            scheduler_params=self.scheduler_params,
            early_stopping_metric=self.metric,
            is_batch_level=False,
        )

        # Reset any previous calls
        self.mock_scheduler.step.reset_mock()

        callback.on_epoch_end(0, self.test_logs)

        # Verify scheduler.step() was called with metric value
        self.mock_scheduler.step.assert_called_once_with(self.test_logs[self.metric])

    def test_on_epoch_end_batch_level(self):
        """Test on_epoch_end for batch-level scheduler (should do nothing)."""
        # Remove is_better attribute if it exists from previous tests
        if hasattr(self.mock_scheduler_fn, "is_better"):
            delattr(self.mock_scheduler_fn, "is_better")

        callback = LRSchedulerCallback(
            scheduler_fn=self.mock_scheduler_fn,
            optimizer=self.mock_optimizer,
            scheduler_params=self.scheduler_params,
            early_stopping_metric=self.metric,
            is_batch_level=True,
        )

        # Reset any previous calls
        self.mock_scheduler.step.reset_mock()

        callback.on_epoch_end(0, self.test_logs)

        # Verify scheduler.step() was not called again
        self.mock_scheduler.step.assert_not_called()

    def test_on_epoch_end_missing_metric(self):
        """Test on_epoch_end when metric is missing from logs."""
        # Remove is_better attribute if it exists from previous tests
        if hasattr(self.mock_scheduler_fn, "is_better"):
            delattr(self.mock_scheduler_fn, "is_better")

        callback = LRSchedulerCallback(
            scheduler_fn=self.mock_scheduler_fn,
            optimizer=self.mock_optimizer,
            scheduler_params=self.scheduler_params,
            early_stopping_metric=self.metric,
            is_batch_level=False,
        )

        # Reset any previous calls
        self.mock_scheduler.step.reset_mock()

        # Call with empty logs (missing the metric)
        callback.on_epoch_end(0, {})

        # Verify scheduler.step() was not called
        self.mock_scheduler.step.assert_not_called()
