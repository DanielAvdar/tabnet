"""Tests for unsupervised_metrics.py module."""

import unittest

import torch

from pytorch_tabnet.metrics.unsupervised_metrics import UnsupervisedMetric, UnsupervisedNumpyMetric


class TestUnsupervisedMetric(unittest.TestCase):
    """Tests for the UnsupervisedMetric class."""

    def test_name_and_maximize(self):
        """Test the name and maximize properties."""
        metric = UnsupervisedMetric()
        self.assertEqual(metric._name, "unsup_loss")
        self.assertFalse(metric._maximize)  # Lower is better for loss

    def test_call(self):
        """Test calling the metric."""
        metric = UnsupervisedMetric()

        # Setting up test data
        embedded_x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        y_pred = embedded_x.clone() + 1.0  # Imperfect reconstruction
        obf_vars = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        result = metric(y_pred, embedded_x, obf_vars)

        # Result should be a float
        self.assertIsInstance(result, float)
        # Loss should be positive for imperfect reconstruction
        self.assertGreater(result, 0.0)

    def test_call_with_weights(self):
        """Test calling the metric with weights."""
        metric = UnsupervisedMetric()

        # Setting up test data
        embedded_x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        y_pred = embedded_x.clone() + 1.0  # Imperfect reconstruction
        obf_vars = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        weights = torch.tensor([2.0, 1.0, 0.5])

        weighted_result = metric(y_pred, embedded_x, obf_vars, weights)
        unweighted_result = metric(y_pred, embedded_x, obf_vars)

        # Results should be different
        self.assertNotEqual(weighted_result, unweighted_result)


class TestUnsupervisedNumpyMetric(unittest.TestCase):
    """Tests for the UnsupervisedNumpyMetric class."""

    def test_name_and_maximize(self):
        """Test the name and maximize properties."""
        metric = UnsupervisedNumpyMetric()
        self.assertEqual(metric._name, "unsup_loss_numpy")
        self.assertFalse(metric._maximize)  # Lower is better for loss

    def test_call(self):
        """Test calling the metric."""
        metric = UnsupervisedNumpyMetric()

        # Setting up test data
        embedded_x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        y_pred = embedded_x.clone() + 1.0  # Imperfect reconstruction
        obf_vars = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        result = metric(y_pred, embedded_x, obf_vars)

        # Result should be a float
        self.assertIsInstance(result, float)
        # Loss should be positive for imperfect reconstruction
        self.assertGreater(result, 0.0)

    def test_comparison_with_unsupervised_metric(self):
        """Test that this metric produces the same results as UnsupervisedMetric when no weights are used."""
        numpy_metric = UnsupervisedNumpyMetric()
        standard_metric = UnsupervisedMetric()

        # Setting up test data
        embedded_x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        y_pred = embedded_x.clone() + 1.0  # Imperfect reconstruction
        obf_vars = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        numpy_result = numpy_metric(y_pred, embedded_x, obf_vars)
        standard_result = standard_metric(y_pred, embedded_x, obf_vars)

        # Results should be the same (no weights in either case)
        self.assertAlmostEqual(numpy_result, standard_result, places=5)
