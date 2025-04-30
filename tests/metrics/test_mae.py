"""Tests for mae.py module."""

import unittest

import torch

from pytorch_tabnet.metrics.mae import MAE


class TestMAE(unittest.TestCase):
    """Tests for the Mean Absolute Error metric class."""

    def test_name_and_maximize(self):
        """Test the name and maximize properties."""
        metric = MAE()
        self.assertEqual(metric._name, "mae")
        self.assertFalse(metric._maximize)  # Lower is better for MAE

    def test_call_perfect_score(self):
        """Test perfect prediction case."""
        metric = MAE()
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        score = metric(y_true, y_pred)
        self.assertEqual(score, 0.0)  # Perfect predictions should have MAE of 0

    def test_call_imperfect_score(self):
        """Test imperfect prediction case."""
        metric = MAE()
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([2.0, 1.0, 4.0])
        score = metric(y_true, y_pred)
        # Expected MAE: (|2-1| + |1-2| + |4-3|) / 3 = (1 + 1 + 1) / 3 = 1.0
        self.assertEqual(score, 1.0)

    def test_call_with_weights(self):
        """Test with sample weights."""
        metric = MAE()
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([2.0, 1.0, 4.0])

        # Apply weights to the errors
        weights = torch.tensor([2.0, 1.0, 2.0])

        weighted_score = metric(y_true, y_pred, weights)
        # Note: The implementation multiplies errors by weights but doesn't
        # normalize by the sum of weights, so the result is actually the mean
        # of the weighted errors.
        # Expected: mean(2*|2-1| + 1*|1-2| + 2*|4-3|) = mean([2, 1, 2]) = 5/3 ≈ 1.67
        self.assertAlmostEqual(weighted_score, 5 / 3, places=5)

    def test_negative_values(self):
        """Test with negative values."""
        metric = MAE()
        y_true = torch.tensor([-1.0, -2.0, 3.0])
        y_pred = torch.tensor([-2.0, -1.0, 2.0])
        score = metric(y_true, y_pred)
        # Expected MAE: (|-2-(-1)| + |-1-(-2)| + |2-3|) / 3 = (|-1| + |1| + |-1|) / 3 = 1.0
        self.assertEqual(score, 1.0)

    def test_2d_arrays(self):
        """Test with 2D arrays for multivariate regression."""
        metric = MAE()
        y_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y_pred = torch.tensor([[1.5, 1.5], [2.5, 4.5]])
        score = metric(y_true, y_pred)
        # Expected MAE: (|1.5-1| + |1.5-2| + |2.5-3| + |4.5-4|) / 4 = (0.5 + 0.5 + 0.5 + 0.5) / 4 = 0.5
        self.assertEqual(score, 0.5)
