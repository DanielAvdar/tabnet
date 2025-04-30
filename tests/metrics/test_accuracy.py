"""Tests for accuracy.py module."""

import unittest

import torch

from pytorch_tabnet.metrics.accuracy import Accuracy


class TestAccuracy(unittest.TestCase):
    """Tests for the Accuracy metric class."""

    def test_name_and_maximize(self):
        """Test the name and maximize properties."""
        metric = Accuracy()
        self.assertEqual(metric._name, "accuracy")
        self.assertTrue(metric._maximize)

    def test_call_perfect_score(self):
        """Test perfect prediction case."""
        metric = Accuracy()
        y_true = torch.tensor([0, 1, 2])
        y_pred = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        score = metric(y_true, y_pred)
        self.assertEqual(score, 1.0)

    def test_call_imperfect_score(self):
        """Test imperfect prediction case."""
        metric = Accuracy()
        y_true = torch.tensor([0, 1, 2])
        y_pred = torch.tensor([
            [0.0, 1.0, 0.0],  # incorrect
            [0.0, 1.0, 0.0],  # correct
            [0.0, 0.0, 1.0],  # correct
        ])
        score = metric(y_true, y_pred)
        self.assertAlmostEqual(score, 2 / 3, places=5)

    def test_call_with_weights(self):
        """Test with sample weights."""
        metric = Accuracy()
        y_true = torch.tensor([0, 1, 2])
        y_pred = torch.tensor([
            [1.0, 0.0, 0.0],  # correct
            [0.0, 1.0, 0.0],  # correct
            [1.0, 0.0, 0.0],  # incorrect
        ])
        # Weights don't actually affect accuracy calculation for now
        weights = torch.tensor([1.0, 1.0, 1.0])
        score = metric(y_true, y_pred, weights)
        self.assertAlmostEqual(score, 2 / 3, places=5)
