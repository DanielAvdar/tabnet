"""Tests for the SparsePredictDataset class."""

import unittest

import numpy as np
import scipy.sparse
import torch

from pytorch_tabnet.data_handlers.sparse_predict_dataset import SparsePredictDataset


class TestSparsePredictDataset(unittest.TestCase):
    """Test cases for SparsePredictDataset class."""

    def test_init(self):
        """Test initialization with sparse matrix."""
        # Create a sparse matrix
        x_dense = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.float32)
        x_sparse = scipy.sparse.csr_matrix(x_dense)

        # Initialize the dataset
        dataset = SparsePredictDataset(x_sparse)

        # Check that data was correctly converted to PyTorch tensor
        self.assertIsInstance(dataset.x, torch.Tensor)

        # Check the shape of tensor
        self.assertEqual(dataset.x.shape, (3, 3))

        # Check values are preserved
        np.testing.assert_allclose(dataset.x.numpy(), x_dense)

    def test_len(self):
        """Test the __len__ method."""
        # Create a sparse matrix with 5 samples
        data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        row = np.array([0, 1, 2, 3, 4])
        col = np.array([0, 1, 2, 1, 0])
        x_sparse = scipy.sparse.csr_matrix((data, (row, col)), shape=(5, 3))

        dataset = SparsePredictDataset(x_sparse)

        # Check length
        self.assertEqual(len(dataset), 5)

    def test_getitem(self):
        """Test the __getitem__ method."""
        # Create a sparse matrix
        x_dense = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.float32)
        x_sparse = scipy.sparse.csr_matrix(x_dense)

        dataset = SparsePredictDataset(x_sparse)

        # Get an item by index
        x_item = dataset[1]

        # Check type
        self.assertIsInstance(x_item, torch.Tensor)

        # Check values
        np.testing.assert_allclose(x_item.numpy(), np.array([0, 2, 0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
