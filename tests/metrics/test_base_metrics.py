"""Tests for base_metrics.py module."""

import unittest
from unittest.mock import MagicMock, patch

import pytest
import torch

from pytorch_tabnet.metrics.base_metrics import (
    Metric,
    MetricContainer,
    UnsupMetricContainer,
    check_metrics,
)


class TestBaseMetrics(unittest.TestCase):
    """Tests for the base Metric class."""

    def test_abstract_call(self):
        """Test that the abstract __call__ method raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            metric = Metric()
            metric(torch.tensor([0]), torch.tensor([0]))

    def test_get_metrics_by_names(self):
        """Test getting metrics by their names."""

        # Create a subclass of Metric for testing
        class TestMetric(Metric):
            _name = "test_metric"
            _maximize = True

            def __call__(self, y_true, y_pred, weights=None):
                return 0.0

        # Add to global list of subclasses (hack for testing)
        orig_subclasses = Metric.__subclasses__()
        try:
            Metric.__subclasses__ = lambda: orig_subclasses + [TestMetric]

            metrics = Metric.get_metrics_by_names(["test_metric"])
            assert len(metrics) == 1
            assert metrics[0]._name == "test_metric"

            # Test with invalid metric name
            with pytest.raises(AssertionError):
                Metric.get_metrics_by_names(["invalid_metric"])
        finally:
            # Restore original subclasses method
            Metric.__subclasses__ = lambda: orig_subclasses


class TestMetricContainer(unittest.TestCase):
    """Tests for the MetricContainer class."""

    def test_init(self):
        """Test initialization of MetricContainer."""

        # Mock a metric class
        class TestMetric(Metric):
            _name = "test_metric"
            _maximize = True

            def __call__(self, y_true, y_pred, weights=None):
                return 0.5

        # Test with default prefix
        with patch("pytorch_tabnet.metrics.base_metrics.Metric.get_metrics_by_names", return_value=[TestMetric()]):
            container = MetricContainer(metric_names=["test_metric"])
            assert container.prefix == ""
            assert container.names == ["test_metric"]

            # Test with custom prefix
            container = MetricContainer(metric_names=["test_metric"], prefix="val_")
            assert container.prefix == "val_"
            assert container.names == ["val_test_metric"]

    def test_call(self):
        """Test calling the MetricContainer."""
        # Create a simple test metric
        test_metric = MagicMock()
        test_metric._name = "test_metric"
        test_metric.return_value = 0.5

        # Mock the get_metrics_by_names to return our test metric
        with patch("pytorch_tabnet.metrics.base_metrics.Metric.get_metrics_by_names", return_value=[test_metric]):
            container = MetricContainer(metric_names=["test_metric"])
            y_true = torch.tensor([[0, 1], [1, 0]])
            y_pred = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
            result = container(y_true, y_pred)

            # Verify the metric was called and the result is correct
            test_metric.assert_called_once()
            assert "test_metric" in result
            assert result["test_metric"] == 0.5

        # Test with list of predictions
        y_pred_list = [torch.tensor([0.1, 0.8]), torch.tensor([0.9, 0.2])]
        with patch("pytorch_tabnet.metrics.base_metrics.Metric.get_metrics_by_names", return_value=[test_metric]):
            container = MetricContainer(metric_names=["test_metric"])
            result = container(y_true, y_pred_list)
            assert "test_metric" in result


class TestUnsupMetricContainer(unittest.TestCase):
    """Tests for the UnsupMetricContainer class."""

    def test_init(self):
        """Test initialization of UnsupMetricContainer."""
        # Mock a metric class
        test_unsup_metric = MagicMock()
        test_unsup_metric._name = "test_unsup_metric"

        # Test with default prefix
        with patch("pytorch_tabnet.metrics.base_metrics.Metric.get_metrics_by_names", return_value=[test_unsup_metric]):
            container = UnsupMetricContainer(metric_names=["test_unsup_metric"])
            assert container.prefix == ""
            assert container.names == ["test_unsup_metric"]

            # Test with custom prefix
            container = UnsupMetricContainer(metric_names=["test_unsup_metric"], prefix="val_")
            assert container.prefix == "val_"
            assert container.names == ["val_test_unsup_metric"]

    def test_call(self):
        """Test calling the UnsupMetricContainer."""
        # Create a simple test metric
        test_unsup_metric = MagicMock()
        test_unsup_metric._name = "test_unsup_metric"
        test_unsup_metric.return_value = 0.5

        # Mock the get_metrics_by_names to return our test metric
        with patch("pytorch_tabnet.metrics.base_metrics.Metric.get_metrics_by_names", return_value=[test_unsup_metric]):
            container = UnsupMetricContainer(metric_names=["test_unsup_metric"])
            y_pred = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
            embedded_x = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
            obf_vars = torch.tensor([[1, 0], [0, 1]])
            result = container(y_pred, embedded_x, obf_vars)

            # Verify the metric was called and the result is correct
            test_unsup_metric.assert_called_once()
            assert "test_unsup_metric" in result
            assert result["test_unsup_metric"] == 0.5


class TestCheckMetrics(unittest.TestCase):
    """Tests for the check_metrics function."""

    def test_string_metrics(self):
        """Test check_metrics with string inputs."""
        metrics = ["metric1", "metric2"]
        result = check_metrics(metrics)
        assert result == metrics

    def test_class_metrics(self):
        """Test check_metrics with Metric subclass inputs."""

        class TestMetric(Metric):
            _name = "test_metric"
            _maximize = True

            def __call__(self, y_true, y_pred, weights=None):
                return 0.5

        result = check_metrics([TestMetric])
        assert result == ["test_metric"]

    def test_invalid_metrics(self):
        """Test check_metrics with invalid inputs."""
        with pytest.raises(TypeError):
            check_metrics([123])  # Not a string or Metric subclass

    def test_invalid_object_metric(self):
        """Test check_metrics with an object that is neither a string nor a Metric subclass."""

        # Use a non-class object that will trigger the else clause
        class NotAMetric:
            pass

        # Since we're passing an instance rather than a class, it will fail at issubclass
        # with TypeError: issubclass() arg 1 must be a class
        with pytest.raises(TypeError):
            check_metrics([NotAMetric()])

    def test_invalid_class_metric(self):
        """Test check_metrics with a class that's not a Metric subclass."""

        # This class should trigger the else clause and the custom error message
        class NotAMetric:
            pass

        with pytest.raises(TypeError, match="You need to provide a valid metric format"):
            check_metrics([NotAMetric])
