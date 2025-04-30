"""Tests for mse.py module."""

import unittest

import torch

from pytorch_tabnet.metrics.mse import MSE


class TestMSE(unittest.TestCase):
    """Tests for the Mean Squared Error metric class."""

    def test_name_and_maximize(self):
        """Test the name and maximize properties."""
        metric = MSE()
        self.assertEqual(metric._name, "mse")
        self.assertFalse(metric._maximize)  # Lower is better for MSE

    def test_call_perfect_score(self):
        """Test perfect prediction case."""
        metric = MSE()
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        score = metric(y_true, y_pred)
        self.assertEqual(score, 0.0)  # Perfect predictions should have MSE of 0

    def test_call_imperfect_score(self):
        """Test imperfect prediction case."""
        metric = MSE()
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([2.0, 1.0, 4.0])
        score = metric(y_true, y_pred)
        # Expected MSE: ((2-1)^2 + (1-2)^2 + (4-3)^2) / 3 = (1 + 1 + 1) / 3 = 1.0
        self.assertEqual(score, 1.0)

    def test_call_with_weights(self):
        """Test with sample weights."""
        metric = MSE()
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([2.0, 1.0, 4.0])

        # Apply weights to the errors
        weights = torch.tensor([2.0, 1.0, 2.0])

        weighted_score = metric(y_true, y_pred, weights)
        # Note: The implementation multiplies squared errors by weights but doesn't
        # normalize by the sum of weights, so the result is actually the mean
        # of the weighted squared errors.
        # Expected: mean(2*(2-1)^2 + 1*(1-2)^2 + 2*(4-3)^2) = mean([2, 1, 2]) = 5/3 ≈ 1.67
        self.assertAlmostEqual(weighted_score, 5 / 3, places=5)

    def test_2d_arrays(self):
        """Test with 2D arrays for multivariate regression."""
        metric = MSE()
        y_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y_pred = torch.tensor([[1.5, 1.5], [2.5, 4.5]])
        score = metric(y_true, y_pred)
        # Expected MSE: ((1.5-1)^2 + (1.5-2)^2 + (2.5-3)^2 + (4.5-4)^2) / 4 = (0.25 + 0.25 + 0.25 + 0.25) / 4 = 0.25
        self.assertEqual(score, 0.25)
