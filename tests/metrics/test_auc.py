"""Tests for auc.py module."""

import unittest

import torch

from pytorch_tabnet.metrics.auc import AUC


class TestAUC(unittest.TestCase):
    """Tests for the AUC metric class."""

    def test_name_and_maximize(self):
        """Test the name and maximize properties."""
        metric = AUC()
        self.assertEqual(metric._name, "auc")
        self.assertTrue(metric._maximize)

    def test_call_binary_case(self):
        """Test AUC calculation for binary classification."""
        metric = AUC()
        y_true = torch.tensor([0, 1, 0, 1])
        y_pred = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]])
        score = metric(y_true, y_pred)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        # For perfect predictions, AUC should be 1.0
        y_pred_perfect = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        score_perfect = metric(y_true, y_pred_perfect)
        self.assertAlmostEqual(score_perfect, 1.0, places=5)

    def test_call_multiclass_case(self):
        """Test AUC calculation for multiclass classification."""
        metric = AUC()
        y_true = torch.tensor([0, 1, 2, 0])
        y_pred = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.2, 0.6], [0.7, 0.2, 0.1]])
        score = metric(y_true, y_pred)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_call_with_weights(self):
        """Test with sample weights."""
        metric = AUC()
        y_true = torch.tensor([0, 1, 0, 1])
        y_pred = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]])
        # Weights don't actually affect AUC calculation in current implementation
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0])
        score = metric(y_true, y_pred, weights)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
