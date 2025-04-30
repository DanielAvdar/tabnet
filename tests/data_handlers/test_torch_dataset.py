"""Tests for the TorchDataset class."""

import unittest

import numpy as np
import torch

from pytorch_tabnet.data_handlers.torch_dataset import TorchDataset


class TestTorchDataset(unittest.TestCase):
    """Test cases for TorchDataset class."""

    def test_init(self):
        """Test initialization of TorchDataset."""
        # Create sample data
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        y = np.array([[0, 1], [1, 0], [0, 1]], dtype=np.float32)

        # Initialize the dataset
        dataset = TorchDataset(x, y)

        # Check that data was correctly converted to PyTorch tensors
        self.assertIsInstance(dataset.x, torch.Tensor)
        self.assertIsInstance(dataset.y, torch.Tensor)

        # Check the shape of tensors
        self.assertEqual(dataset.x.shape, (3, 3))
        self.assertEqual(dataset.y.shape, (3, 2))

        # Check values are preserved
        np.testing.assert_allclose(dataset.x.numpy(), x)
        np.testing.assert_allclose(dataset.y.numpy(), y)

    def test_len(self):
        """Test the __len__ method."""
        # Create a dataset with 5 samples
        x = np.random.rand(5, 10).astype(np.float32)
        y = np.random.rand(5, 2).astype(np.float32)
        dataset = TorchDataset(x, y)

        # Check length
        self.assertEqual(len(dataset), 5)

    def test_getitem(self):
        """Test the __getitem__ method."""
        # Create sample data
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        y = np.array([[0, 1], [1, 0], [0, 1]], dtype=np.float32)
        dataset = TorchDataset(x, y)

        # Get an item by index
        x_item, y_item = dataset[1]

        # Check types
        self.assertIsInstance(x_item, torch.Tensor)
        self.assertIsInstance(y_item, torch.Tensor)

        # Check values
        np.testing.assert_allclose(x_item.numpy(), np.array([4, 5, 6], dtype=np.float32))
        np.testing.assert_allclose(y_item.numpy(), np.array([1, 0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
