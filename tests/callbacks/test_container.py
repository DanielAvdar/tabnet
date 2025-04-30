from unittest.mock import MagicMock, patch

from pytorch_tabnet.callbacks import Callback, CallbackContainer


class TestCallbackContainer:
    """Test the CallbackContainer class functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.container = CallbackContainer()
        self.mock_callback1 = MagicMock(spec=Callback)
        self.mock_callback2 = MagicMock(spec=Callback)
        self.test_params = {"param1": 1, "param2": "test"}
        self.test_logs = {"loss": 0.5, "accuracy": 0.9}
        self.mock_trainer = MagicMock()

    def test_init(self):
        """Test initialization of CallbackContainer."""
        container = CallbackContainer()
        assert isinstance(container, CallbackContainer)
        assert container.callbacks == []

        # Test with initial callbacks
        callbacks = [self.mock_callback1, self.mock_callback2]
        container = CallbackContainer(callbacks=callbacks)
        assert container.callbacks == callbacks

    def test_append(self):
        """Test appending a callback."""
        self.container.append(self.mock_callback1)
        assert len(self.container.callbacks) == 1
        assert self.container.callbacks[0] == self.mock_callback1

        self.container.append(self.mock_callback2)
        assert len(self.container.callbacks) == 2
        assert self.container.callbacks[1] == self.mock_callback2

    def test_set_params(self):
        """Test setting parameters for all callbacks."""
        self.container.append(self.mock_callback1)
        self.container.append(self.mock_callback2)

        self.container.set_params(self.test_params)

        self.mock_callback1.set_params.assert_called_once_with(self.test_params)
        self.mock_callback2.set_params.assert_called_once_with(self.test_params)

    def test_set_trainer(self):
        """Test setting trainer for all callbacks."""
        self.container.append(self.mock_callback1)
        self.container.append(self.mock_callback2)

        self.container.set_trainer(self.mock_trainer)

        assert hasattr(self.container, "trainer")
        assert self.container.trainer == self.mock_trainer
        self.mock_callback1.set_trainer.assert_called_once_with(self.mock_trainer)
        self.mock_callback2.set_trainer.assert_called_once_with(self.mock_trainer)

    def test_on_epoch_begin(self):
        """Test on_epoch_begin calling all callbacks."""
        self.container.append(self.mock_callback1)
        self.container.append(self.mock_callback2)

        # Test with logs
        self.container.on_epoch_begin(1, self.test_logs)
        self.mock_callback1.on_epoch_begin.assert_called_once_with(1, self.test_logs)
        self.mock_callback2.on_epoch_begin.assert_called_once_with(1, self.test_logs)

        # Reset mocks
        self.mock_callback1.reset_mock()
        self.mock_callback2.reset_mock()

        # Test without logs
        self.container.on_epoch_begin(2)
        self.mock_callback1.on_epoch_begin.assert_called_once_with(2, {})
        self.mock_callback2.on_epoch_begin.assert_called_once_with(2, {})

    def test_on_epoch_end(self):
        """Test on_epoch_end calling all callbacks."""
        self.container.append(self.mock_callback1)
        self.container.append(self.mock_callback2)

        # Test with logs
        self.container.on_epoch_end(1, self.test_logs)
        self.mock_callback1.on_epoch_end.assert_called_once_with(1, self.test_logs)
        self.mock_callback2.on_epoch_end.assert_called_once_with(1, self.test_logs)

        # Reset mocks
        self.mock_callback1.reset_mock()
        self.mock_callback2.reset_mock()

        # Test without logs
        self.container.on_epoch_end(2)
        self.mock_callback1.on_epoch_end.assert_called_once_with(2, {})
        self.mock_callback2.on_epoch_end.assert_called_once_with(2, {})

    def test_on_batch_begin(self):
        """Test on_batch_begin calling all callbacks."""
        self.container.append(self.mock_callback1)
        self.container.append(self.mock_callback2)

        # Test with logs
        self.container.on_batch_begin(1, self.test_logs)
        self.mock_callback1.on_batch_begin.assert_called_once_with(1, self.test_logs)
        self.mock_callback2.on_batch_begin.assert_called_once_with(1, self.test_logs)

        # Reset mocks
        self.mock_callback1.reset_mock()
        self.mock_callback2.reset_mock()

        # Test without logs
        self.container.on_batch_begin(2)
        self.mock_callback1.on_batch_begin.assert_called_once_with(2, {})
        self.mock_callback2.on_batch_begin.assert_called_once_with(2, {})

    def test_on_batch_end(self):
        """Test on_batch_end calling all callbacks."""
        self.container.append(self.mock_callback1)
        self.container.append(self.mock_callback2)

        # Test with logs
        self.container.on_batch_end(1, self.test_logs)
        self.mock_callback1.on_batch_end.assert_called_once_with(1, self.test_logs)
        self.mock_callback2.on_batch_end.assert_called_once_with(1, self.test_logs)

        # Reset mocks
        self.mock_callback1.reset_mock()
        self.mock_callback2.reset_mock()

        # Test without logs
        self.container.on_batch_end(2)
        self.mock_callback1.on_batch_end.assert_called_once_with(2, {})
        self.mock_callback2.on_batch_end.assert_called_once_with(2, {})

    @patch("time.time", return_value=12345)
    def test_on_train_begin(self, mock_time):
        """Test on_train_begin calling all callbacks with start time."""
        self.container.append(self.mock_callback1)
        self.container.append(self.mock_callback2)

        # Test with logs
        expected_logs = {**self.test_logs, "start_time": 12345}
        self.container.on_train_begin(self.test_logs)
        self.mock_callback1.on_train_begin.assert_called_once_with(expected_logs)
        self.mock_callback2.on_train_begin.assert_called_once_with(expected_logs)

        # Reset mocks
        self.mock_callback1.reset_mock()
        self.mock_callback2.reset_mock()

        # Test without logs
        expected_logs = {"start_time": 12345}
        self.container.on_train_begin()
        self.mock_callback1.on_train_begin.assert_called_once_with(expected_logs)
        self.mock_callback2.on_train_begin.assert_called_once_with(expected_logs)

    def test_on_train_end(self):
        """Test on_train_end calling all callbacks."""
        self.container.append(self.mock_callback1)
        self.container.append(self.mock_callback2)

        # Test with logs
        self.container.on_train_end(self.test_logs)
        self.mock_callback1.on_train_end.assert_called_once_with(self.test_logs)
        self.mock_callback2.on_train_end.assert_called_once_with(self.test_logs)

        # Reset mocks
        self.mock_callback1.reset_mock()
        self.mock_callback2.reset_mock()

        # Test without logs
        self.container.on_train_end()
        self.mock_callback1.on_train_end.assert_called_once_with({})
        self.mock_callback2.on_train_end.assert_called_once_with({})
