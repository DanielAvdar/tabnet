"""Tests for the metrics package initialization."""

import unittest

from pytorch_tabnet.metrics import (
    AUC,
    MAE,
    MSE,
    RMSE,
    RMSLE,
    Accuracy,
    BalancedAccuracy,
    LogLoss,
    Metric,
    MetricContainer,
    UnsupervisedLoss,
    UnsupervisedMetric,
    UnsupervisedNumpyMetric,
    UnsupMetricContainer,
    check_metrics,
)


class TestMetricsInit(unittest.TestCase):
    """Tests for the metrics package initialization."""

    def test_import_all_metrics(self):
        """Test that all metrics can be imported from the package."""
        # Verify the metrics are imported correctly
        self.assertIsNotNone(Accuracy)
        self.assertIsNotNone(AUC)
        self.assertIsNotNone(BalancedAccuracy)
        self.assertIsNotNone(Metric)
        self.assertIsNotNone(MetricContainer)
        self.assertIsNotNone(UnsupMetricContainer)
        self.assertIsNotNone(check_metrics)
        self.assertIsNotNone(LogLoss)
        self.assertIsNotNone(MAE)
        self.assertIsNotNone(MSE)
        self.assertIsNotNone(RMSE)
        self.assertIsNotNone(RMSLE)
        self.assertIsNotNone(UnsupervisedLoss)
        self.assertIsNotNone(UnsupervisedMetric)
        self.assertIsNotNone(UnsupervisedNumpyMetric)

    def test_metric_instances(self):
        """Test creating instances of all metric classes."""
        # Create instances of each metric class
        metrics = [
            Accuracy(),
            AUC(),
            BalancedAccuracy(),
            LogLoss(),
            MAE(),
            MSE(),
            RMSE(),
            RMSLE(),
            UnsupervisedMetric(),
            UnsupervisedNumpyMetric(),
        ]

        # Verify each metric has the required attributes
        for metric in metrics:
            self.assertTrue(hasattr(metric, "_name"))
            self.assertTrue(hasattr(metric, "_maximize"))

    def test_get_metrics_by_names(self):
        """Test getting metrics by names using the Metric.get_metrics_by_names method."""
        metric_names = ["accuracy", "auc", "balanced_accuracy", "logloss", "mae", "mse", "rmse", "rmsle"]

        metrics = Metric.get_metrics_by_names(metric_names)

        # Check that all requested metrics were retrieved
        self.assertEqual(len(metrics), len(metric_names))
        retrieved_names = [metric._name for metric in metrics]
        for name in metric_names:
            self.assertIn(name, retrieved_names)
