from unittest.mock import MagicMock

from pytorch_tabnet.callbacks import Callback


class TestCallback:
    """Test the base Callback class functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.callback = Callback()
        self.mock_model = MagicMock()
        self.test_params = {"param1": 1, "param2": "test"}
        self.test_logs = {"loss": 0.5, "accuracy": 0.9}

    def test_init(self):
        """Test initialization of Callback."""
        callback = Callback()
        assert isinstance(callback, Callback)

    def test_set_params(self):
        """Test setting parameters."""
        self.callback.set_params(self.test_params)
        assert hasattr(self.callback, "params")
        assert self.callback.params == self.test_params

    def test_set_trainer(self):
        """Test setting trainer/model."""
        self.callback.set_trainer(self.mock_model)
        assert hasattr(self.callback, "trainer")
        assert self.callback.trainer == self.mock_model

    def test_on_epoch_begin(self):
        """Test on_epoch_begin method."""
        # Should not raise any exceptions
        self.callback.on_epoch_begin(0)
        self.callback.on_epoch_begin(1, self.test_logs)

    def test_on_epoch_end(self):
        """Test on_epoch_end method."""
        # Should not raise any exceptions
        self.callback.on_epoch_end(0)
        self.callback.on_epoch_end(1, self.test_logs)

    def test_on_batch_begin(self):
        """Test on_batch_begin method."""
        # Should not raise any exceptions
        self.callback.on_batch_begin(0)
        self.callback.on_batch_begin(1, self.test_logs)

    def test_on_batch_end(self):
        """Test on_batch_end method."""
        # Should not raise any exceptions
        self.callback.on_batch_end(0)
        self.callback.on_batch_end(1, self.test_logs)

    def test_on_train_begin(self):
        """Test on_train_begin method."""
        # Should not raise any exceptions
        self.callback.on_train_begin()
        self.callback.on_train_begin(self.test_logs)

    def test_on_train_end(self):
        """Test on_train_end method."""
        # Should not raise any exceptions
        self.callback.on_train_end()
        self.callback.on_train_end(self.test_logs)
