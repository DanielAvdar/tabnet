import pytest
import torch

# import numpy as np
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


def test_UnsupervisedLoss():
    y_pred = torch.randn(3, 5)
    embedded_x = torch.randn(3, 5)
    obf_vars = torch.randint(0, 2, (3, 5)).float()
    loss = UnsupervisedLoss(y_pred, embedded_x, obf_vars)
    assert loss.item() >= 0

    # Test with all zeros in obf_vars
    obf_vars = torch.zeros(3, 5)
    loss = UnsupervisedLoss(y_pred, embedded_x, obf_vars)
    assert loss.item() == 0


def test_UnsupMetricContainer():
    container = UnsupMetricContainer(metric_names=["unsup_loss"])
    y_pred = torch.randn(3, 5)
    embedded_x = torch.randn(3, 5)
    obf_vars = torch.randint(0, 2, (3, 5)).float()

    logs = container(y_pred, embedded_x, obf_vars)
    assert "unsup_loss" in logs


def test_MetricContainer():
    container = MetricContainer(metric_names=["auc"])

    y_true = torch.randint(0, 2, (100,))
    y_pred = torch.rand(100, 2)
    logs = container(y_true, y_pred)
    assert "auc" in logs


def test_check_metrics():
    metrics = check_metrics(["auc", "accuracy", MAE, RMSE])
    assert "auc" in metrics and "accuracy" in metrics and "mae" in metrics and "rmse" in metrics
    with pytest.raises(TypeError):
        check_metrics([1])


#####################


def test_UnsupervisedLoss_edge_cases():
    # Test with empty tensors
    y_pred = torch.empty(0, 5)
    embedded_x = torch.empty(0, 5)
    obf_vars = torch.empty(0, 5)
    loss = UnsupervisedLoss(y_pred, embedded_x, obf_vars)
    assert torch.isnan(loss)  # Expecting NaN for empty tensors


@pytest.mark.parametrize("metric_name", ["unsup_loss"])
def test_UnsupMetricContainer_different_metrics(metric_name):
    container = UnsupMetricContainer(metric_names=[metric_name])
    y_pred = torch.randn(3, 5)
    embedded_x = torch.randn(3, 5)
    obf_vars = torch.randint(0, 2, (3, 5)).float()

    logs = container(y_pred, embedded_x, obf_vars)
    assert metric_name in logs


@pytest.mark.parametrize(
    "metric_name",
    ["auc", "accuracy", "balanced_accuracy", "logloss", "mae", "mse", "rmsle", "rmse"],
)
def test_MetricContainer_different_metrics(metric_name):
    container = MetricContainer(metric_names=[metric_name])

    if metric_name not in ["auc", "accuracy", "logloss", "balanced_accuracy"]:
        y_true = torch.rand(100, 2)
        y_pred = torch.rand(100, 2)
        _ = container(y_true, y_pred)
    else:
        y_true = torch.randint(0, 2, (100,))
        y_pred = torch.rand(100, 2)
        _ = container(y_true, y_pred)


def test_Metric_get_metrics_by_names():
    metrics = Metric.get_metrics_by_names(["auc", "accuracy"])
    assert len(metrics) == 2
    assert isinstance(metrics[0], AUC)
    assert isinstance(metrics[1], Accuracy)

    with pytest.raises(AssertionError):
        Metric.get_metrics_by_names(["non_existent_metric"])


@pytest.mark.parametrize(
    "metric_cls, y_true, y_score, expected",
    [
        (
            AUC,
            torch.tensor([0, 1, 1, 0]),
            torch.tensor([
                [0.1, 0.9],
                [0.2, 0.8],
                [0.3, 0.7],
                [0.4, 0.6],
            ]),
            0.5,
        ),
        (
            Accuracy,
            torch.tensor([0, 1, 1, 0]),
            torch.tensor([
                [0.9, 0.1],
                [0.2, 0.8],
                [0.3, 0.7],
                [0.6, 0.4],
            ]),
            1,
        ),
        (
            BalancedAccuracy,
            torch.tensor([0, 1, 1, 0]),
            torch.tensor([
                [0.9, 0.1],
                [0.2, 0.8],
                [0.3, 0.7],
                [0.6, 0.4],
            ]),
            1,
        ),
        (
            LogLoss,
            torch.tensor([0, 1, 1, 0]),
            torch.tensor([
                [0.9, 0.1],
                [0.2, 0.8],
                [0.3, 0.7],
                [0.6, 0.4],
            ]),
            0.48,
        ),
        (
            MAE,
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.5, 2.5, 3.5]),
            0.5,
        ),
        (
            MSE,
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.5, 2.5, 3.5]),
            0.25,
        ),
        (
            RMSLE,
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.5, 2.5, 3.5]),
            0.17,
        ),
        (
            RMSE,
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.5, 2.5, 3.5]),
            0.5,
        ),
    ],
)
def test_metric_calculations(metric_cls, y_true, y_score, expected):
    metric = metric_cls()
    if metric._name == "logloss":
        result = metric(y_true, y_score)
        assert abs(result - expected) < 0.001

    elif metric._name in ["mae", "mse", "rmsle", "rmse"]:
        result = metric(y_true, y_score)
        assert abs(result - expected) < 0.002
    else:
        result = metric(y_true, y_score)
        assert result == expected


def test_UnsupervisedMetric():
    metric = UnsupervisedMetric()
    y_pred = torch.randn(3, 5)
    embedded_x = torch.randn(3, 5)
    obf_vars = torch.randint(0, 2, (3, 5)).float()
    loss = metric(y_pred, embedded_x, obf_vars)
    assert loss >= 0


def test_UnsupervisedNumpyMetric():
    metric = UnsupervisedNumpyMetric()

    y_pred = torch.rand(3, 5)
    embedded_x = torch.rand(3, 5)
    obf_vars = torch.randint(0, 2, (3, 5)).float()
    loss = metric(y_pred, embedded_x, obf_vars)
    assert loss >= 0
