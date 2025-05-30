"""Tests for logloss.py module."""

import unittest

import torch

from pytorch_tabnet.metrics.logloss import LogLoss


class TestLogLoss(unittest.TestCase):
    """Tests for the LogLoss metric class."""

    def test_name_and_maximize(self):
        """Test the name and maximize properties."""
        metric = LogLoss()
        self.assertEqual(metric._name, "logloss")
        self.assertFalse(metric._maximize)  # Lower is better for logloss

    def test_call_perfect_score(self):
        """Test perfect prediction case."""
        metric = LogLoss()
        y_true = torch.tensor([0, 1, 2])
        y_pred = torch.tensor(
            [
                [100.0, 0.0, 0.0],  # Very confident correct prediction
                [0.0, 100.0, 0.0],  # Very confident correct prediction
                [0.0, 0.0, 100.0],  # Very confident correct prediction
            ]
        )
        score = metric(y_true, y_pred)
        # Perfect predictions should have a very low logloss
        self.assertLess(score, 0.01)

    def test_call_imperfect_score(self):
        """Test imperfect prediction case."""
        metric = LogLoss()
        y_true = torch.tensor([0, 1, 2])
        y_pred = torch.tensor(
            [
                [0.5, 0.3, 0.2],  # Uncertain prediction
                [0.1, 0.5, 0.4],  # Uncertain prediction
                [0.2, 0.3, 0.5],  # Uncertain prediction
            ]
        )
        score = metric(y_true, y_pred)
        # Uncertain predictions should have higher logloss
        self.assertGreater(score, 0.5)

    def test_call_with_weights(self):
        """Test with sample weights."""
        metric = LogLoss()
        y_true = torch.tensor([0, 1, 2])

        # First sample has confident wrong prediction, others are correct
        y_pred = torch.tensor(
            [
                [0.0, 1.0, 0.0],  # Wrong prediction for class 0
                [0.0, 1.0, 0.0],  # Correct prediction for class 1
                [0.0, 0.0, 1.0],  # Correct prediction for class 2
            ]
        )

        # Higher weight on the incorrect prediction
        weights = torch.tensor([2.0, 1.0, 1.0])

        weighted_score = metric(y_true, y_pred, weights)
        unweighted_score = metric(y_true, y_pred)

        # Weighted score should be higher due to higher weight on incorrect prediction
        self.assertGreater(weighted_score, unweighted_score)
