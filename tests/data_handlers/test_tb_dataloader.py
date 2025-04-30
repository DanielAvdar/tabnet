"""Tests for the TBDataLoader class."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from pytorch_tabnet.data_handlers.predict_dataset import PredictDataset
from pytorch_tabnet.data_handlers.tb_dataloader import TBDataLoader
from pytorch_tabnet.data_handlers.torch_dataset import TorchDataset


class TestTBDataLoader(unittest.TestCase):
    """Test cases for TBDataLoader class."""

    def test_init(self):
        """Test initialization of TBDataLoader."""
        # Create a mock dataset
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        y = np.array([[0, 1], [1, 0], [0, 1]], dtype=np.float32)
        dataset = TorchDataset(x, y)

        # Create a dataloader
        dataloader = TBDataLoader(
            name="test_loader",
            dataset=dataset,
            batch_size=2,
            weights=None,
            pre_training=False,
            drop_last=False,
            pin_memory=False,
            predict=False,
            all_at_once=False,
        )

        # Check attributes
        self.assertEqual(dataloader.name, "test_loader")
        self.assertEqual(dataloader.batch_size, 2)
        self.assertFalse(dataloader.pre_training)
        self.assertFalse(dataloader.drop_last)
        self.assertFalse(dataloader.pin_memory)
        self.assertFalse(dataloader.predict)
        self.assertFalse(dataloader.all_at_once)

    def test_len(self):
        """Test the __len__ method."""
        # Test with normal batch size
        x = np.random.rand(10, 5).astype(np.float32)
        y = np.random.rand(10, 2).astype(np.float32)
        dataset = TorchDataset(x, y)

        dataloader = TBDataLoader(name="test_loader", dataset=dataset, batch_size=3, predict=False)
        # Expected length is ceil(10/3) = 4
        self.assertEqual(len(dataloader), 4)

        # Test with drop_last=True
        dataloader = TBDataLoader(name="test_loader", dataset=dataset, batch_size=3, drop_last=True, predict=False)
        # Expected length is ceil(10/3) - 1 = 3
        self.assertEqual(len(dataloader), 3)

        # Test with predict=True and drop_last=True (drop_last should be ignored)
        dataloader = TBDataLoader(name="test_loader", dataset=dataset, batch_size=3, drop_last=True, predict=True)
        # Expected length is ceil(10/3) = 4
        self.assertEqual(len(dataloader), 4)

    def test_get_weights(self):
        """Test the get_weights method."""
        # Create dataset with weights
        x = np.random.rand(5, 3).astype(np.float32)
        y = np.random.rand(5, 2).astype(np.float32)
        dataset = TorchDataset(x, y)
        weights = torch.tensor([0.5, 1.0, 0.7, 0.3, 0.8])

        dataloader = TBDataLoader(name="test_loader", dataset=dataset, batch_size=2, weights=weights)

        # Test get_weights without index
        returned_weights = dataloader.get_weights()
        torch.testing.assert_close(returned_weights, weights)

        # Test get_weights with index
        indexes = torch.tensor([1, 3])
        returned_weights = dataloader.get_weights(indexes)
        torch.testing.assert_close(returned_weights, torch.tensor([1.0, 0.3]))

        # Test get_weights with no weights
        dataloader = TBDataLoader(name="test_loader", dataset=dataset, batch_size=2, weights=None)
        self.assertIsNone(dataloader.get_weights())
        self.assertIsNone(dataloader.get_weights(indexes))

    def test_iter_all_at_once_predict(self):
        """Test __iter__ with all_at_once=True for predict dataset."""
        # Create predict dataset
        x = np.random.rand(5, 3).astype(np.float32)
        dataset = PredictDataset(x)

        dataloader = TBDataLoader(name="test_loader", dataset=dataset, batch_size=2, all_at_once=True)

        # Get the batch
        batch_iter = iter(dataloader)
        x_batch, y_batch, w_batch = next(batch_iter)

        # Check that we got all samples at once
        self.assertEqual(x_batch.shape[0], 5)
        self.assertIsNone(y_batch)
        self.assertIsNone(w_batch)

        # Check that there are no more batches
        with self.assertRaises(StopIteration):
            next(batch_iter)

    def test_iter_all_at_once_train(self):
        """Test __iter__ with all_at_once=True for torch dataset."""
        # Create torch dataset
        x = np.random.rand(5, 3).astype(np.float32)
        y = np.random.rand(5, 2).astype(np.float32)
        dataset = TorchDataset(x, y)

        dataloader = TBDataLoader(name="test_loader", dataset=dataset, batch_size=2, all_at_once=True)

        # Get the batch
        batch_iter = iter(dataloader)
        x_batch, y_batch, w_batch = next(batch_iter)

        # Check that we got all samples at once
        self.assertEqual(x_batch.shape[0], 5)
        self.assertEqual(y_batch.shape[0], 5)
        self.assertIsNone(w_batch)

        # Check that there are no more batches
        with self.assertRaises(StopIteration):
            next(batch_iter)

    def test_make_predict_batch(self):
        """Test the make_predict_batch method."""
        # Create dataset for prediction
        x = np.random.rand(10, 3).astype(np.float32)
        dataset = PredictDataset(x)

        dataloader = TBDataLoader(name="test_loader", dataset=dataset, batch_size=4, predict=True)

        # Test normal batch
        x_batch, y_batch, w_batch = dataloader.make_predict_batch(10, 0)
        self.assertEqual(x_batch.shape[0], 4)
        self.assertIsNone(y_batch)
        self.assertIsNone(w_batch)

        # Test last batch (smaller than batch_size)
        x_batch, y_batch, w_batch = dataloader.make_predict_batch(10, 8)
        self.assertEqual(x_batch.shape[0], 2)
        self.assertIsNone(y_batch)
        self.assertIsNone(w_batch)

        # Test with weights
        weights = torch.ones(10)
        dataloader = TBDataLoader(name="test_loader", dataset=dataset, batch_size=4, predict=True, weights=weights)

        x_batch, y_batch, w_batch = dataloader.make_predict_batch(10, 4)
        self.assertEqual(x_batch.shape[0], 4)
        self.assertIsNone(y_batch)
        self.assertEqual(w_batch.shape[0], 4)

    def test_make_train_batch(self):
        """Test the make_train_batch method."""
        # Create dataset for training
        x = np.random.rand(10, 3).astype(np.float32)
        y = np.random.rand(10, 2).astype(np.float32)
        dataset = TorchDataset(x, y)

        dataloader = TBDataLoader(name="test_loader", dataset=dataset, batch_size=4)

        # Create a permutation of indices
        perm = torch.randperm(10)

        # Test normal batch
        x_batch, y_batch, w_batch = dataloader.make_train_batch(10, perm, 0)
        self.assertEqual(x_batch.shape[0], 4)
        self.assertEqual(y_batch.shape[0], 4)
        self.assertIsNone(w_batch)

        # Test with weights
        weights = torch.ones(10)
        dataloader = TBDataLoader(name="test_loader", dataset=dataset, batch_size=4, weights=weights)

        x_batch, y_batch, w_batch = dataloader.make_train_batch(10, perm, 4)
        self.assertEqual(x_batch.shape[0], 4)
        self.assertEqual(y_batch.shape[0], 4)
        self.assertEqual(w_batch.shape[0], 4)

        # Test with leftover (when batch goes beyond dataset size)
        x_batch, y_batch, w_batch = dataloader.make_train_batch(10, perm, 8)
        self.assertEqual(x_batch.shape[0], 4)  # 2 from end + 2 from beginning
        self.assertEqual(y_batch.shape[0], 4)
        self.assertEqual(w_batch.shape[0], 4)

    def test_make_train_batch_pretraining(self):
        """Test make_train_batch with pre_training=True."""
        # Create dataset for pre-training
        x = np.random.rand(10, 3).astype(np.float32)
        y = np.random.rand(10, 2).astype(np.float32)
        dataset = TorchDataset(x, y)

        dataloader = TBDataLoader(name="test_loader", dataset=dataset, batch_size=4, pre_training=True)

        # Create a permutation of indices
        perm = torch.randperm(10)

        # Test normal batch in pre-training mode
        x_batch, y_batch, w_batch = dataloader.make_train_batch(10, perm, 0)
        self.assertEqual(x_batch.shape[0], 4)
        self.assertIsNone(y_batch)  # y should be None in pre-training mode
        self.assertIsNone(w_batch)

        # Test with leftover in pre-training mode
        x_batch, y_batch, w_batch = dataloader.make_train_batch(10, perm, 8)
        self.assertEqual(x_batch.shape[0], 4)  # 2 from end + 2 from beginning
        self.assertIsNone(y_batch)
        self.assertIsNone(w_batch)

    def test_iter_predict_mode(self):
        """Test iteration in predict mode."""
        # Create dataset
        x = np.random.rand(5, 3).astype(np.float32)
        dataset = PredictDataset(x)

        dataloader = TBDataLoader(name="test_loader", dataset=dataset, batch_size=2, predict=True)

        # Collect all batches
        batches = list(dataloader)

        # Check number of batches
        self.assertEqual(len(batches), 3)  # ceil(5/2) = 3

        # Check sizes of batches
        self.assertEqual(batches[0][0].shape[0], 2)
        self.assertEqual(batches[1][0].shape[0], 2)
        self.assertEqual(batches[2][0].shape[0], 1)  # Last batch has only 1 sample

    def test_iter_train_mode(self):
        """Test iteration in training mode."""
        # Create dataset
        x = np.random.rand(5, 3).astype(np.float32)
        y = np.random.rand(5, 2).astype(np.float32)
        dataset = TorchDataset(x, y)

        # Mock torch.randperm to return a fixed permutation for deterministic testing
        with patch("torch.randperm", return_value=torch.tensor([0, 1, 2, 3, 4])):
            dataloader = TBDataLoader(name="test_loader", dataset=dataset, batch_size=2)

            # Collect all batches
            batches = list(dataloader)

            # Check number of batches
            self.assertEqual(len(batches), 3)  # ceil(5/2) = 3

            # Check sizes of batches
            self.assertEqual(batches[0][0].shape[0], 2)
            self.assertEqual(batches[1][0].shape[0], 2)
            # After observing the implementation, we see the last batch is also size 2
            # (1 from end + 1 from beginning) due to how leftover handling works
            self.assertEqual(batches[2][0].shape[0], 2)  # The implementation pads the last batch to full size

    def test_make_predict_batch_with_torch_dataset(self):
        """Test the make_predict_batch method with a regular TorchDataset."""
        # Create a TorchDataset (not PredictDataset or SparsePredictDataset)
        x = np.random.rand(10, 3).astype(np.float32)
        y = np.random.rand(10, 2).astype(np.float32)
        dataset = TorchDataset(x, y)

        dataloader = TBDataLoader(
            name="test_loader",
            dataset=dataset,
            batch_size=4,
            predict=True,  # Important: set predict=True
            pre_training=False,  # Make sure pre_training is False
        )

        # Test batch using make_predict_batch
        x_batch, y_batch, w_batch = dataloader.make_predict_batch(10, 4)

        # Check batch sizes
        self.assertEqual(x_batch.shape[0], 4)
        self.assertEqual(y_batch.shape[0], 4)  # Now y is not None
        self.assertIsNone(w_batch)


if __name__ == "__main__":
    unittest.main()
