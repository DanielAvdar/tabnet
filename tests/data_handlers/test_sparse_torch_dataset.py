"""Tests for the SparseTorchDataset class."""

import unittest

import numpy as np
import torch

from pytorch_tabnet.data_handlers.sparse_torch_dataset import SparseTorchDataset


class TestSparseTorchDataset(unittest.TestCase):
    """Test cases for SparseTorchDataset class."""

    def test_init(self):
        """Test initialization with dense matrix and labels."""
        # Create a dense array for features
        x_dense = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.float32)
        x_sparse = x_dense

        # Create labels
        y = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)

        # Initialize the dataset
        dataset = SparseTorchDataset(x_sparse, y)

        # Check that data was correctly converted to PyTorch tensors
        self.assertIsInstance(dataset.x, torch.Tensor)
        self.assertIsInstance(dataset.y, torch.Tensor)

        # Check the shapes of tensors
        self.assertEqual(dataset.x.shape, (3, 3))
        self.assertEqual(dataset.y.shape, (3, 2))

        # Check values are preserved
        np.testing.assert_allclose(dataset.x.numpy(), x_dense)
        np.testing.assert_allclose(dataset.y.numpy(), y)

    def test_len(self):
        """Test the __len__ method."""
        # Create a matrix with 5 samples
        data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        row = np.array([0, 1, 2, 3, 4])
        col = np.array([0, 1, 2, 1, 0])
        x_sparse = np.zeros((5, 3), dtype=np.float32)
        for i in range(len(data)):
            x_sparse[row[i], col[i]] = data[i]

        # Create labels
        y = np.random.rand(5, 2).astype(np.float32)

        dataset = SparseTorchDataset(x_sparse, y)

        # Check length
        self.assertEqual(len(dataset), 5)

    def test_getitem(self):
        """Test the __getitem__ method."""
        # Create a dense array
        x_dense = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.float32)
        x_sparse = x_dense

        # Create labels
        y = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)

        dataset = SparseTorchDataset(x_sparse, y)

        # Get an item by index
        x_item, y_item = dataset[1]

        # Check types
        self.assertIsInstance(x_item, torch.Tensor)
        self.assertIsInstance(y_item, torch.Tensor)

        # Check values
        np.testing.assert_allclose(x_item.numpy(), np.array([0, 2, 0], dtype=np.float32))
        np.testing.assert_allclose(y_item.numpy(), np.array([0, 1], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
