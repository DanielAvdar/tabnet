import torch

from pytorch_tabnet.utils.device import define_device


class TestDevice:
    """Tests for the device utility functions."""

    def test_define_device_auto_with_cuda(self, monkeypatch):
        """Test define_device with 'auto' when CUDA is available."""
        # Mock torch.cuda.is_available to return True
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        assert define_device("auto") == "cuda"

    def test_define_device_auto_without_cuda(self, monkeypatch):
        """Test define_device with 'auto' when CUDA is not available."""
        # Mock torch.cuda.is_available to return False
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        assert define_device("auto") == "cpu"

    def test_define_device_cuda_not_available(self, monkeypatch):
        """Test define_device with 'cuda' when CUDA is not available."""
        # Mock torch.cuda.is_available to return False
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        assert define_device("cuda") == "cpu"

    def test_define_device_cuda_available(self, monkeypatch):
        """Test define_device with 'cuda' when CUDA is available."""
        # Mock torch.cuda.is_available to return True
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        assert define_device("cuda") == "cuda"

    def test_define_device_cpu(self):
        """Test define_device with 'cpu'."""
        assert define_device("cpu") == "cpu"

    def test_define_device_other(self):
        """Test define_device with an arbitrary value."""
        assert define_device("other_device") == "other_device"
