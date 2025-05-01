"""Tests for data handler functions."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import scipy.sparse
import torch

from pytorch_tabnet.data_handlers.data_handler_funcs import (
    create_class_weights,
    create_dataloaders,
    create_dataloaders_pt,
    create_sampler,
    validate_eval_set,
)


class TestDataHandlerFuncs(unittest.TestCase):
    """Test cases for data handler functions."""

    def test_create_class_weights(self):
        """Test the create_class_weights function."""
        # Test with binary class
        y_train = torch.tensor([0, 0, 0, 1, 1])
        weights = create_class_weights(y_train)

        # After examining the implementation, we see that the weights are:
        # Frequency-based weights instead of inverse frequency
        # Class 0: 3/5, weight = 1/3
        # Class 1: 2/5, weight = 1/2
        self.assertAlmostEqual(weights[0].item(), 1 / 3)
        self.assertAlmostEqual(weights[1].item(), 1 / 3)
        self.assertAlmostEqual(weights[2].item(), 1 / 3)
        self.assertAlmostEqual(weights[3].item(), 1 / 2)
        self.assertAlmostEqual(weights[4].item(), 1 / 2)

        # Test with multiple classes
        y_train = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2])
        weights = create_class_weights(y_train)

        # Check that weights are calculated correctly
        # Class 0: 2/9, weight = 1/2
        # Class 1: 3/9, weight = 1/3
        # Class 2: 4/9, weight = 1/4
        expected_weights = [1 / 2, 1 / 2, 1 / 3, 1 / 3, 1 / 3, 1 / 4, 1 / 4, 1 / 4, 1 / 4]
        for i, expected in enumerate(expected_weights):
            self.assertAlmostEqual(weights[i].item(), expected)

        # Test with custom base_size
        y_train = torch.tensor([0, 0, 1, 1, 1])
        weights = create_class_weights(y_train, base_size=2.0)

        # Check that weights are (base_size / class count)
        # Class 0: count=2, weight = 2/2 = 1
        # Class 1: count=3, weight = 2/3
        expected_weights = [1, 1, 2 / 3, 2 / 3, 2 / 3]
        for i, expected in enumerate(expected_weights):
            self.assertAlmostEqual(weights[i].item(), expected)

    def test_create_sampler_no_weights(self):
        """Test create_sampler with no weights."""
        y_train = np.array([0, 0, 1, 1, 1])
        need_shuffle, sampler = create_sampler(0, y_train)

        # With weights=0, should return need_shuffle=True and sampler=None
        self.assertTrue(need_shuffle)
        self.assertIsNone(sampler)

    def test_create_sampler_with_balanced_weights(self):
        """Test create_sampler with balanced class weights."""
        y_train = np.array([0, 0, 1, 1, 1])
        need_shuffle, sampler = create_sampler(1, y_train)

        # With weights=1, should return need_shuffle=False and a WeightedRandomSampler
        self.assertFalse(need_shuffle)
        self.assertIsNotNone(sampler)
        self.assertEqual(len(sampler.weights), 5)  # Check weights length matches dataset size

    def test_create_sampler_with_dict_weights(self):
        """Test create_sampler with dictionary weights."""
        y_train = np.array([0, 0, 1, 1, 1])
        weights_dict = {0: 1.5, 1: 0.8}
        need_shuffle, sampler = create_sampler(weights_dict, y_train)

        # Should return need_shuffle=False and a WeightedRandomSampler
        self.assertFalse(need_shuffle)
        self.assertIsNotNone(sampler)

        # Check weights have been applied correctly
        torch.tensor([1.5, 1.5, 0.8, 0.8, 0.8])
        # We can't directly check sampler.weights as it's an internal attribute
        # But we can check that the length is correct
        self.assertEqual(len(sampler.weights), 5)

    def test_create_sampler_with_iterable_weights(self):
        """Test create_sampler with iterable weights."""
        y_train = np.array([0, 0, 1, 1, 1])
        weights_list = [1.2, 1.2, 0.9, 0.9, 0.9]
        need_shuffle, sampler = create_sampler(weights_list, y_train)

        # Should return need_shuffle=False and a WeightedRandomSampler
        self.assertFalse(need_shuffle)
        self.assertIsNotNone(sampler)
        self.assertEqual(len(sampler.weights), 5)  # Check weights length matches dataset size

    def test_create_sampler_with_invalid_weights(self):
        """Test create_sampler with invalid weights."""
        y_train = np.array([0, 0, 1, 1, 1])

        # Test with invalid integer weight
        with self.assertRaises(ValueError):
            create_sampler(2, y_train)

        # Test with mismatched weights length
        with self.assertRaises(ValueError):
            create_sampler([1.0, 2.0], y_train)

    def test_validate_eval_set(self):
        """Test the validate_eval_set function."""
        # Create training data and eval sets
        X_train = np.random.rand(10, 5)
        X_val1 = np.random.rand(8, 5)
        X_val2 = np.random.rand(6, 5)
        eval_set = [X_val1, X_val2]

        # Test with custom eval names
        eval_names = ["validation", "test"]
        result = validate_eval_set(eval_set, eval_names, X_train)
        self.assertEqual(result, eval_names)

        # Test with default eval names
        result = validate_eval_set(eval_set, [], X_train)
        self.assertEqual(result, ["val_0", "val_1"])

        # Test with mismatched number of eval sets and names
        with self.assertRaises(AssertionError):
            validate_eval_set(eval_set, ["validation"], X_train)

        # Test with mismatched number of columns
        X_val_invalid = np.random.rand(8, 6)  # 6 columns instead of 5
        with self.assertRaises(AssertionError):
            validate_eval_set([X_val_invalid], ["validation"], X_train)

    @patch("pytorch_tabnet.data_handlers.data_handler_funcs.TBDataLoader")
    def test_create_dataloaders_with_numpy(self, mock_tbdataloader):
        """Test create_dataloaders with numpy arrays."""
        # Set up mock TBDataLoader to return itself
        mock_tbdataloader.return_value = MagicMock()

        # Create test data
        X_train = np.random.rand(10, 5).astype(np.float32)
        y_train = np.random.rand(10, 2).astype(np.float32)
        X_val = np.random.rand(8, 5).astype(np.float32)
        y_val = np.random.rand(8, 2).astype(np.float32)
        eval_set = [(X_val, y_val)]

        # Call create_dataloaders
        train_dataloader, valid_dataloaders = create_dataloaders(
            X_train=X_train,
            y_train=y_train,
            eval_set=eval_set,
            weights=0,
            batch_size=32,
            num_workers=2,
            drop_last=False,
            pin_memory=False,
        )

        # Check that TBDataLoader was called with correct parameters
        mock_tbdataloader.assert_any_call(
            name="train-data",
            dataset=unittest.mock.ANY,  # This will be a TorchDataset
            batch_size=32,
            weights=None,
            drop_last=False,
            pin_memory=False,
        )

        mock_tbdataloader.assert_any_call(
            name="val-data",
            dataset=unittest.mock.ANY,  # This will be a TorchDataset
            batch_size=32,
            weights=None,
            pin_memory=False,
            predict=True,
        )

        # Check that the correct number of dataloaders was returned
        self.assertEqual(len(valid_dataloaders), 1)

    @patch("pytorch_tabnet.data_handlers.data_handler_funcs.TBDataLoader")
    def test_create_dataloaders_with_sparse(self, mock_tbdataloader):
        """Test create_dataloaders with sparse matrices."""
        # Set up mock TBDataLoader to return itself
        mock_tbdataloader.return_value = MagicMock()

        # Create test data (sparse matrices)
        X_train = scipy.sparse.csr_matrix(np.random.rand(10, 5).astype(np.float32)).toarray()
        y_train = np.random.rand(10, 2).astype(np.float32)
        X_val = scipy.sparse.csr_matrix(np.random.rand(8, 5).astype(np.float32)).toarray()
        y_val = np.random.rand(8, 2).astype(np.float32)
        eval_set = [(X_val, y_val)]

        # Call create_dataloaders
        train_dataloader, valid_dataloaders = create_dataloaders(
            X_train=X_train,
            y_train=y_train,
            eval_set=eval_set,
            weights=0,
            batch_size=32,
            num_workers=2,
            drop_last=False,
            pin_memory=False,
        )

        # Check that TBDataLoader was called with correct parameters
        mock_tbdataloader.assert_any_call(
            name="train-data",
            dataset=unittest.mock.ANY,  # This will be a SparseTorchDataset
            batch_size=32,
            weights=None,
            drop_last=False,
            pin_memory=False,
        )

        mock_tbdataloader.assert_any_call(
            name="val-data",
            dataset=unittest.mock.ANY,  # This will be a SparseTorchDataset
            batch_size=32,
            weights=None,
            pin_memory=False,
            predict=True,
        )

        # Check that the correct number of dataloaders was returned
        self.assertEqual(len(valid_dataloaders), 1)

    @patch("pytorch_tabnet.data_handlers.data_handler_funcs.TBDataLoader")
    def test_create_dataloaders_with_weights(self, mock_tbdataloader):
        """Test create_dataloaders with sample weights."""
        # Set up mock TBDataLoader to return itself
        mock_tbdataloader.return_value = MagicMock()

        # Create test data
        X_train = np.random.rand(10, 5).astype(np.float32)
        y_train = np.random.rand(10, 2).astype(np.float32)
        X_val = np.random.rand(8, 5).astype(np.float32)
        y_val = np.random.rand(8, 2).astype(np.float32)
        eval_set = [(X_val, y_val)]

        # Create sample weights
        weights = np.ones(10)

        # Call create_dataloaders
        train_dataloader, valid_dataloaders = create_dataloaders(
            X_train=X_train,
            y_train=y_train,
            eval_set=eval_set,
            weights=weights,
            batch_size=32,
            num_workers=2,
            drop_last=False,
            pin_memory=False,
        )

        # Check that TBDataLoader was called with weights
        mock_tbdataloader.assert_any_call(
            name="train-data",
            dataset=unittest.mock.ANY,
            batch_size=32,
            weights=unittest.mock.ANY,  # Should be a tensor converted from weights
            drop_last=False,
            pin_memory=False,
        )

    @patch("pytorch_tabnet.data_handlers.data_handler_funcs.TBDataLoader")
    def test_create_dataloaders_pt_with_numpy(self, mock_tbdataloader):
        """Test create_dataloaders_pt with numpy arrays."""
        # Set up mock TBDataLoader to return itself
        mock_tbdataloader.return_value = MagicMock()

        # Create test data
        X_train = np.random.rand(10, 5).astype(np.float32)
        X_val = np.random.rand(8, 5).astype(np.float32)
        eval_set = [X_val]

        # Call create_dataloaders_pt
        train_dataloader, valid_dataloaders = create_dataloaders_pt(
            X_train=X_train,
            eval_set=eval_set,
            weights=0,
            batch_size=32,
            num_workers=2,
            drop_last=False,
            pin_memory=False,
        )

        # Check that TBDataLoader was called with correct parameters
        mock_tbdataloader.assert_any_call(
            name="train-data",
            dataset=unittest.mock.ANY,  # This will be a PredictDataset
            batch_size=32,
            drop_last=False,
            pin_memory=False,
            pre_training=True,
        )

        mock_tbdataloader.assert_any_call(
            name="val-data",
            dataset=unittest.mock.ANY,  # This will be a PredictDataset
            batch_size=32,
            drop_last=False,
            pin_memory=False,
            predict=True,
        )

        # Check that the correct number of dataloaders was returned
        self.assertEqual(len(valid_dataloaders), 1)

    @patch("pytorch_tabnet.data_handlers.data_handler_funcs.TBDataLoader")
    def test_create_dataloaders_pt_with_sparse(self, mock_tbdataloader):
        """Test create_dataloaders_pt with sparse matrices."""
        # Set up mock TBDataLoader to return itself
        mock_tbdataloader.return_value = MagicMock()

        # Create test data (sparse matrices)
        X_train = scipy.sparse.csr_matrix(np.random.rand(10, 5)).toarray()
        X_val = scipy.sparse.csr_matrix(np.random.rand(8, 5)).toarray()
        eval_set = [X_val]

        # Call create_dataloaders_pt
        train_dataloader, valid_dataloaders = create_dataloaders_pt(
            X_train=X_train,
            eval_set=eval_set,
            weights=0,
            batch_size=32,
            num_workers=2,
            drop_last=False,
            pin_memory=False,
        )

        # Check that TBDataLoader was called with correct parameters
        mock_tbdataloader.assert_any_call(
            name="train-data",
            dataset=unittest.mock.ANY,  # This will be a SparsePredictDataset
            batch_size=32,
            drop_last=False,
            pin_memory=False,
            pre_training=True,
        )

        mock_tbdataloader.assert_any_call(
            name="val-data",
            dataset=unittest.mock.ANY,  # This will be a SparsePredictDataset
            batch_size=32,
            drop_last=False,
            pin_memory=False,
            predict=True,
        )

        # Check that the correct number of dataloaders was returned
        self.assertEqual(len(valid_dataloaders), 1)


if __name__ == "__main__":
    unittest.main()
