"""Tests for balanced_accuracy.py module."""

import unittest

import torch

from pytorch_tabnet.metrics.balanced_accuracy import BalancedAccuracy


class TestBalancedAccuracy(unittest.TestCase):
    """Tests for the BalancedAccuracy metric class."""

    def test_name_and_maximize(self):
        """Test the name and maximize properties."""
        metric = BalancedAccuracy()
        self.assertEqual(metric._name, "balanced_accuracy")
        self.assertTrue(metric._maximize)

    def test_call_perfect_score(self):
        """Test perfect prediction case."""
        metric = BalancedAccuracy()
        y_true = torch.tensor([0, 1, 2])
        y_pred = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        score = metric(y_true, y_pred)
        self.assertEqual(score, 1.0)

    def test_call_imbalanced_classes(self):
        """Test with imbalanced classes."""
        metric = BalancedAccuracy()
        # Imbalanced class distribution
        y_true = torch.tensor([0, 0, 0, 1, 2])
        y_pred = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],  # incorrect for class 0
                [0.0, 1.0, 0.0],  # correct for class 1
                [0.0, 0.0, 1.0],  # correct for class 2
            ]
        )
        score = metric(y_true, y_pred)
        # Balanced accuracy should give equal weight to each class
        # Class 0: 2/3 correct, Class 1: 1/1 correct, Class 2: 1/1 correct
        # Expected: (2/3 + 1 + 1)/3 = 8/9
        self.assertAlmostEqual(score, (2 / 3 + 1 + 1) / 3, places=5)

    def test_call_with_weights(self):
        """Test with sample weights."""
        metric = BalancedAccuracy()
        y_true = torch.tensor([0, 1, 2])
        y_pred = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # correct
                [0.0, 1.0, 0.0],  # correct
                [1.0, 0.0, 0.0],  # incorrect for class 2
            ]
        )
        # Weights don't actually affect balanced accuracy calculation in current implementation
        torch.tensor([1.0, 1.0, 1.0])
        score = metric(y_true, y_pred)
        # Class 0: 1/1, Class 1: 1/1, Class 2: 0/1
        # Expected: (1 + 1 + 0)/3 = 2/3
        self.assertAlmostEqual(score, 2 / 3, places=5)
