"""Tests for the data_types module."""

import unittest

import numpy as np
import torch

from pytorch_tabnet.data_handlers.data_types import X_type, tn_type


class TestDataTypes(unittest.TestCase):
    """Test cases for data types."""

    def test_x_type_numpy(self):
        """Test X_type with numpy array."""
        x = np.array([[1, 2], [3, 4]])
        self.assertIsInstance(x, X_type)

    def test_tn_type_tensor(self):
        """Test tn_type with tensor."""
        x = torch.tensor([[1, 2], [3, 4]])
        self.assertIsInstance(x, tn_type)

    def test_tn_type_none(self):
        """Test tn_type with None."""
        x = None
        self.assertIsInstance(x, tn_type)


if __name__ == "__main__":
    unittest.main()
