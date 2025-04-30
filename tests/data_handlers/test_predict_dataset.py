"""Tests for the PredictDataset class."""

import unittest

import numpy as np
import scipy.sparse
import torch

from pytorch_tabnet.data_handlers.predict_dataset import PredictDataset


class TestPredictDataset(unittest.TestCase):
    """Test cases for PredictDataset class."""

    def test_init_with_numpy_array(self):
        """Test initialization with NumPy array."""
        # Create sample data
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

        # Initialize the dataset
        dataset = PredictDataset(x)

        # Check that data was correctly converted to PyTorch tensor
        self.assertIsInstance(dataset.x, torch.Tensor)

        # Check the shape of tensor
        self.assertEqual(dataset.x.shape, (3, 3))

        # Check values are preserved
        np.testing.assert_allclose(dataset.x.numpy(), x)

    def test_init_with_sparse_matrix(self):
        """Test initialization with SciPy sparse matrix."""
        # Create a sparse matrix
        x_dense = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.float32)
        x_sparse = scipy.sparse.csr_matrix(x_dense)

        # Initialize the dataset
        dataset = PredictDataset(x_sparse)

        # Check that data was correctly converted to PyTorch tensor
        self.assertIsInstance(dataset.x, torch.Tensor)

        # Check the shape of tensor
        self.assertEqual(dataset.x.shape, (3, 3))

        # Check values are preserved
        np.testing.assert_allclose(dataset.x.numpy(), x_dense)

    def test_init_with_torch_tensor(self):
        """Test initialization with PyTorch tensor."""
        # Create a PyTorch tensor
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Initialize the dataset
        dataset = PredictDataset(x)

        # Check that data remains a PyTorch tensor
        self.assertIsInstance(dataset.x, torch.Tensor)

        # Check the shape of tensor
        self.assertEqual(dataset.x.shape, (2, 3))

        # Check values are preserved
        torch.testing.assert_close(dataset.x, x.float())

    def test_len(self):
        """Test the __len__ method."""
        # Create a dataset with 5 samples
        x = np.random.rand(5, 10).astype(np.float32)
        dataset = PredictDataset(x)

        # Check length
        self.assertEqual(len(dataset), 5)

    def test_getitem(self):
        """Test the __getitem__ method."""
        # Create sample data
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        dataset = PredictDataset(x)

        # Get an item by index
        x_item = dataset[1]

        # Check type
        self.assertIsInstance(x_item, torch.Tensor)

        # Check values
        np.testing.assert_allclose(x_item.numpy(), np.array([4, 5, 6], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
