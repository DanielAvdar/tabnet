from dataclasses import dataclass
from typing import List, Union, Any
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    log_loss,
    balanced_accuracy_score,
    mean_squared_log_error,
)
import torch


def UnsupervisedLoss(
    y_pred: torch.Tensor,
    embedded_x: torch.Tensor,
    obf_vars: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Implements unsupervised loss function.
    This differs from orginal paper as it's scaled to be batch size independent
    and number of features reconstructed independent (by taking the mean)

    Parameters
    ----------
    y_pred : torch.Tensor or np.array
        Reconstructed prediction (with embeddings)
    embedded_x : torch.Tensor
        Original input embedded by network
    obf_vars : torch.Tensor
        Binary mask for obfuscated variables.
        1 means the variable was obfuscated so reconstruction is based on this.
    eps : float
        A small floating point to avoid ZeroDivisionError
        This can happen in degenerated case when a feature has only one value

    Returns
    -------
    loss : torch float
        Unsupervised loss, average value over batch samples.
    """
    errors = y_pred - embedded_x
    reconstruction_errors = torch.mul(errors, obf_vars) ** 2
    batch_means = torch.mean(embedded_x, dim=0)
    batch_means[batch_means == 0] = 1

    batch_stds = torch.std(embedded_x, dim=0) ** 2
    batch_stds[batch_stds == 0] = batch_means[batch_stds == 0]
    features_loss = torch.matmul(reconstruction_errors, 1 / batch_stds)
    # compute the number of obfuscated variables to reconstruct
    nb_reconstructed_variables = torch.sum(obf_vars, dim=1)
    # take the mean of the reconstructed variable errors
    features_loss = features_loss / (nb_reconstructed_variables + eps)
    # here we take the mean per batch, contrary to the paper
    loss = torch.mean(features_loss)
    return loss


def UnsupervisedLossNumpy(  # todo: switch to from numpy to torch
    y_pred: np.ndarray,
    embedded_x: np.ndarray,
    obf_vars: np.ndarray,
    eps: float = 1e-9,
) -> np.ndarray:
    errors = y_pred - embedded_x
    reconstruction_errors = np.multiply(errors, obf_vars) ** 2
    batch_means = np.mean(embedded_x, axis=0)
    batch_means = np.where(batch_means == 0, 1, batch_means)

    batch_stds = np.std(embedded_x, axis=0, ddof=1) ** 2
    batch_stds = np.where(batch_stds == 0, batch_means, batch_stds)
    features_loss = np.matmul(reconstruction_errors, 1 / batch_stds)
    # compute the number of obfuscated variables to reconstruct
    nb_reconstructed_variables = np.sum(obf_vars, axis=1)
    # take the mean of the reconstructed variable errors
    features_loss = features_loss / (nb_reconstructed_variables + eps)
    # here we take the mean per batch, contrary to the paper
    loss = np.mean(features_loss)
    return loss


@dataclass
class UnsupMetricContainer:
    """Container holding a list of metrics.

    Parameters
    ----------
    y_pred : torch.Tensor or np.array
        Reconstructed prediction (with embeddings)
    embedded_x : torch.Tensor
        Original input embedded by network
    obf_vars : torch.Tensor
        Binary mask for obfuscated variables.
        1 means the variables was obfuscated so reconstruction is based on this.

    """

    metric_names: List[str]
    prefix: str = ""

    def __post_init__(self) -> None:
        self.metrics = Metric.get_metrics_by_names(self.metric_names)
        self.names = [self.prefix + name for name in self.metric_names]

    def __call__(  # todo: switch to from numpy to torch
        self,
        y_pred: Union[np.ndarray, torch.Tensor],
        embedded_x: Union[np.ndarray, torch.Tensor],
        obf_vars: Union[np.ndarray, torch.Tensor],
    ) -> dict:
        """Compute all metrics and store into a dict.

        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_pred : np.ndarray
            Score matrix or vector

        Returns
        -------
        dict
            Dict of metrics ({metric_name: metric_value}).

        """
        logs = {}
        for metric in self.metrics:
            res = metric(y_pred, embedded_x, obf_vars)
            logs[self.prefix + metric._name] = res
        return logs


@dataclass
class MetricContainer:
    """Container holding a list of metrics.

    Parameters
    ----------
    metric_names : list of str
        List of metric names.
    prefix : str
        Prefix of metric names.

    """

    metric_names: List[str]
    prefix: str = ""

    def __post_init__(self) -> None:
        self.metrics = Metric.get_metrics_by_names(self.metric_names)
        self.names = [self.prefix + name for name in self.metric_names]

    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict:  # todo: switch to from numpy to torch
        """Compute all metrics and store into a dict.

        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_pred : np.ndarray
            Score matrix or vector

        Returns
        -------
        dict
            Dict of metrics ({metric_name: metric_value}).

        """
        logs = {}
        for metric in self.metrics:
            if isinstance(y_pred, list):
                res = np.mean([
                    metric(y_true[:, i], y_pred[i]) for i in range(len(y_pred))
                ])
            else:
                res = metric(y_true, y_pred)
            logs[self.prefix + metric._name] = res
        return logs


class Metric:
    _name: str
    _maximize: bool

    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:  # todo: switch to from numpy to torch
        raise NotImplementedError("Custom Metrics must implement this function")

    @classmethod
    def get_metrics_by_names(cls, names: List[str]) -> List:
        """Get list of metric classes.

        Parameters
        ----------
        cls : Metric
            Metric class.
        names : list
            List of metric names.

        Returns
        -------
        metrics : list
            List of metric classes.

        """
        available_metrics = cls.__subclasses__()
        available_names = [metric()._name for metric in available_metrics]
        metrics = []
        for name in names:
            assert name in available_names, (
                f"{name} is not available, choose in {available_names}"
            )
            idx = available_names.index(name)
            metric = available_metrics[idx]()
            metrics.append(metric)
        return metrics


class AUC(Metric):
    """
    AUC.
    """

    _name: str = "auc"
    _maximize: bool = True

    def __call__(
        self, y_true: np.ndarray, y_score: np.ndarray
    ) -> float:  # todo: switch to from numpy to torch
        """
        Compute AUC of predictions.

        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_score : np.ndarray
            Score matrix or vector

        Returns
        -------
        float
            AUC of predictions vs targets.
        """
        return roc_auc_score(y_true, y_score[:, 1])


class Accuracy(Metric):
    """
    Accuracy.
    """

    _name: str = "accuracy"
    _maximize: bool = True

    def __call__(
        self, y_true: np.ndarray, y_score: np.ndarray
    ) -> float:  # todo: switch to from numpy to torch
        """
        Compute Accuracy of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_score: np.ndarray
            Score matrix or vector

        Returns
        -------
        float
            Accuracy of predictions vs targets.
        """
        y_pred = np.argmax(y_score, axis=1)
        return accuracy_score(y_true, y_pred)


class BalancedAccuracy(Metric):
    """
    Balanced Accuracy.
    """

    _name: str = "balanced_accuracy"
    _maximize: bool = True

    def __call__(
        self, y_true: np.ndarray, y_score: np.ndarray
    ) -> float:  # todo: switch to from numpy to torch
        """
        Compute Accuracy of predictions.

        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_score : np.ndarray
            Score matrix or vector

        Returns
        -------
        float
            Accuracy of predictions vs targets.
        """
        y_pred = np.argmax(y_score, axis=1)
        return balanced_accuracy_score(y_true, y_pred)


class LogLoss(Metric):
    """
    LogLoss.
    """

    _name: str = "logloss"
    _maximize: bool = False

    def __call__(
        self, y_true: np.ndarray, y_score: np.ndarray
    ) -> float:  # todo: switch to from numpy to torch
        """
        Compute LogLoss of predictions.

        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_score : np.ndarray
            Score matrix or vector

        Returns
        -------
        float
            LogLoss of predictions vs targets.
        """
        return log_loss(y_true, y_score)


class MAE(Metric):
    """
    Mean Absolute Error.
    """

    _name: str = "mae"
    _maximize: bool = False

    def __call__(
        self, y_true: np.ndarray, y_score: np.ndarray
    ) -> float:  # todo: switch to from numpy to torch
        """
        Compute MAE (Mean Absolute Error) of predictions.

        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_score : np.ndarray
            Score matrix or vector

        Returns
        -------
        float
            MAE of predictions vs targets.
        """
        return mean_absolute_error(y_true, y_score)


class MSE(Metric):
    """
    Mean Squared Error.
    """

    _name: str = "mse"
    _maximize: bool = False

    def __call__(
        self, y_true: np.ndarray, y_score: np.ndarray
    ) -> float:  # todo: switch to from numpy to torch
        """
        Compute MSE (Mean Squared Error) of predictions.

        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_score : np.ndarray
            Score matrix or vector

        Returns
        -------
        float
            MSE of predictions vs targets.
        """
        return mean_squared_error(y_true, y_score)


class RMSLE(Metric):
    """
    Root Mean squared logarithmic error regression loss.
    Scikit-implementation:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_log_error.html
    Note: In order to avoid error, negative predictions are clipped to 0.
    This means that you should clip negative predictions manually after calling predict.
    """

    _name: str = "rmsle"
    _maximize: bool = False

    def __call__(
        self, y_true: np.ndarray, y_score: np.ndarray
    ) -> float:  # todo: switch to from numpy to torch
        """
        Compute RMSLE of predictions.

        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_score : np.ndarray
            Score matrix or vector

        Returns
        -------
        float
            RMSLE of predictions vs targets.
        """
        y_score = np.clip(y_score, a_min=0, a_max=None)
        return np.sqrt(mean_squared_log_error(y_true, y_score))


class UnsupervisedMetric(Metric):
    """
    Unsupervised metric
    """

    _name: str = "unsup_loss"
    _maximize: bool = False

    def __call__(  # type: ignore[override]
        self,
        y_pred: Union[np.ndarray, torch.Tensor],
        embedded_x: Union[np.ndarray, torch.Tensor],
        obf_vars: Union[np.ndarray, torch.Tensor],
    ) -> float:  # todo: switch to from numpy to torch
        """
        Compute MSE (Mean Squared Error) of predictions.

        Parameters
        ----------
        y_pred : torch.Tensor or np.array
            Reconstructed prediction (with embeddings)
        embedded_x : torch.Tensor
            Original input embedded by network
        obf_vars : torch.Tensor
            Binary mask for obfuscated variables.
            1 means the variables was obfuscated so reconstruction is based on this.

        Returns
        -------
        float
            MSE of predictions vs targets.
        """
        loss = UnsupervisedLoss(y_pred, embedded_x, obf_vars)
        return loss.item()


class UnsupervisedNumpyMetric(Metric):
    """
    Unsupervised metric
    """

    _name: str = "unsup_loss_numpy"
    _maximize: bool = False

    def __call__(  # type: ignore[override]
        self, y_pred: np.ndarray, embedded_x: np.ndarray, obf_vars: np.ndarray
    ) -> float:  # todo: switch to from numpy to torch

        return UnsupervisedLossNumpy(y_pred, embedded_x, obf_vars)


class RMSE(Metric):
    """
    Root Mean Squared Error.
    """

    _name: str = "rmse"
    _maximize: bool = False

    def __call__(
        self, y_true: np.ndarray, y_score: np.ndarray
    ) -> float:  # todo: switch to from numpy to torch
        """
        Compute RMSE (Root Mean Squared Error) of predictions.

        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_score : np.ndarray
            Score matrix or vector

        Returns
        -------
        float
            RMSE of predictions vs targets.
        """
        return np.sqrt(mean_squared_error(y_true, y_score))


def check_metrics(metrics: List[Union[str, Any]]) -> List[str]:
    """Check if custom metrics are provided.

    Parameters
    ----------
    metrics : list of str or classes
        List with built-in metrics (str) or custom metrics (classes).

    Returns
    -------
    val_metrics : list of str
        List of metric names.

    """
    val_metrics = []
    for metric in metrics:
        if isinstance(metric, str):
            val_metrics.append(metric)
        elif issubclass(metric, Metric):
            val_metrics.append(metric()._name)
        else:
            raise TypeError("You need to provide a valid metric format")
    return val_metrics
