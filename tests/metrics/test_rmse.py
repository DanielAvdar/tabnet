"""Tests for rmse.py module."""

import math
import unittest

import torch

from pytorch_tabnet.metrics.rmse import RMSE


class TestRMSE(unittest.TestCase):
    """Tests for the Root Mean Squared Error metric class."""

    def test_name_and_maximize(self):
        """Test the name and maximize properties."""
        metric = RMSE()
        self.assertEqual(metric._name, "rmse")
        self.assertFalse(metric._maximize)  # Lower is better for RMSE

    def test_call_perfect_score(self):
        """Test perfect prediction case."""
        metric = RMSE()
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        score = metric(y_true, y_pred)
        self.assertEqual(score, 0.0)  # Perfect predictions should have RMSE of 0

    def test_call_imperfect_score(self):
        """Test imperfect prediction case."""
        metric = RMSE()
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([2.0, 1.0, 4.0])
        score = metric(y_true, y_pred)
        # Expected RMSE: sqrt(((2-1)^2 + (1-2)^2 + (4-3)^2) / 3) = sqrt((1 + 1 + 1) / 3) = 1.0
        self.assertEqual(score, 1.0)

    def test_call_with_weights(self):
        """Test with sample weights."""
        metric = RMSE()
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([2.0, 1.0, 4.0])

        # Apply weights to the errors
        weights = torch.tensor([2.0, 1.0, 2.0])

        weighted_score = metric(y_true, y_pred, weights)
        # Note: The implementation multiplies squared errors by weights but doesn't
        # normalize by the sum of weights, so the result is actually the square root
        # of the mean of the weighted squared errors.
        # Expected: sqrt(mean(2*(2-1)^2 + 1*(1-2)^2 + 2*(4-3)^2)) = sqrt(mean([2, 1, 2])) = sqrt(5/3) ≈ 1.29
        self.assertAlmostEqual(weighted_score, math.sqrt(5 / 3), places=5)

    def test_larger_errors(self):
        """Test with larger errors that will produce non-integer RMSE."""
        metric = RMSE()
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([3.0, 1.0, 7.0])
        score = metric(y_true, y_pred)
        # Expected RMSE: sqrt(((3-1)^2 + (1-2)^2 + (7-3)^2) / 3) = sqrt((4 + 1 + 16) / 3) = sqrt(7) ≈ 2.65
        self.assertAlmostEqual(score, math.sqrt(7), places=5)

    def test_2d_arrays(self):
        """Test with 2D arrays for multivariate regression."""
        metric = RMSE()
        y_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y_pred = torch.tensor([[2.0, 3.0], [4.0, 6.0]])
        score = metric(y_true, y_pred)
        # Expected RMSE: sqrt(((2-1)^2 + (3-2)^2 + (4-3)^2 + (6-4)^2) / 4) = sqrt((1 + 1 + 1 + 4) / 4) = sqrt(7/4) ≈ 1.32
        self.assertAlmostEqual(score, math.sqrt(7 / 4), places=5)
