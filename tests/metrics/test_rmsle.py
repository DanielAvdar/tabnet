"""Tests for rmsle.py module."""

import math
import unittest

import torch

from pytorch_tabnet.metrics.rmsle import RMSLE


class TestRMSLE(unittest.TestCase):
    """Tests for the Root Mean Squared Logarithmic Error metric class."""

    def test_name_and_maximize(self):
        """Test the name and maximize properties."""
        metric = RMSLE()
        self.assertEqual(metric._name, "rmsle")
        self.assertFalse(metric._maximize)  # Lower is better for RMSLE

    def test_call_perfect_score(self):
        """Test perfect prediction case."""
        metric = RMSLE()
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        score = metric(y_true, y_pred)
        self.assertEqual(score, 0.0)  # Perfect predictions should have RMSLE of 0

    def test_call_imperfect_score(self):
        """Test imperfect prediction case."""
        metric = RMSLE()
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([2.0, 1.0, 6.0])
        score = metric(y_true, y_pred)
        # Expected RMSLE: sqrt(((log(2+1)-log(1+1))^2 + (log(1+1)-log(2+1))^2 + (log(6+1)-log(3+1))^2) / 3)
        expected = math.sqrt(((math.log(3) - math.log(2)) ** 2 + (math.log(2) - math.log(3)) ** 2 + (math.log(7) - math.log(4)) ** 2) / 3)
        self.assertAlmostEqual(score, expected, places=5)

    def test_call_with_weights(self):
        """Test with sample weights."""
        metric = RMSLE()
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([2.0, 1.0, 6.0])

        # Apply weights to the logarithmic squared errors
        weights = torch.tensor([2.0, 1.0, 2.0])

        weighted_score = metric(y_true, y_pred, weights)
        # Note: The implementation multiplies squared log errors by weights but doesn't
        # normalize by the sum of weights, so the result is actually the square root
        # of the mean of the weighted squared log errors.
        log_err1 = (math.log(3) - math.log(2)) ** 2 * 2.0
        log_err2 = (math.log(2) - math.log(3)) ** 2 * 1.0
        log_err3 = (math.log(7) - math.log(4)) ** 2 * 2.0
        expected = math.sqrt((log_err1 + log_err2 + log_err3) / 3.0)
        self.assertAlmostEqual(weighted_score, expected, places=5)

    def test_zero_values(self):
        """Test with zero values which would be problematic without the +1 term."""
        metric = RMSLE()
        y_true = torch.tensor([0.0, 2.0, 3.0])
        y_pred = torch.tensor([0.0, 1.0, 6.0])
        score = metric(y_true, y_pred)
        # With the +1 term, this should work fine even with zeros
        # Expected RMSLE: sqrt(((log(0+1)-log(0+1))^2 + (log(1+1)-log(2+1))^2 + (log(6+1)-log(3+1))^2) / 3)
        expected = math.sqrt(((math.log(1) - math.log(1)) ** 2 + (math.log(2) - math.log(3)) ** 2 + (math.log(7) - math.log(4)) ** 2) / 3)
        self.assertAlmostEqual(score, expected, places=5)

    def test_2d_arrays(self):
        """Test with 2D arrays for multivariate regression."""
        metric = RMSLE()
        y_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y_pred = torch.tensor([[2.0, 3.0], [4.0, 6.0]])
        score = metric(y_true, y_pred)
        # Expected RMSLE with 2D arrays
        log_err1 = (math.log(3) - math.log(2)) ** 2
        log_err2 = (math.log(4) - math.log(3)) ** 2
        log_err3 = (math.log(5) - math.log(4)) ** 2
        log_err4 = (math.log(7) - math.log(5)) ** 2
        expected = math.sqrt((log_err1 + log_err2 + log_err3 + log_err4) / 4)
        self.assertAlmostEqual(score, expected, places=5)
