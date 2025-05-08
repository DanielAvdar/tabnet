"""TabNet regression model class and training logic."""

from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch

from ..utils import filter_weights
from .abstract_models import TabSupervisedModel


@dataclass
class TabNetRegressor(TabSupervisedModel):
    """TabNet model for regression tasks."""

    output_dim: int = None

    def __post_init__(self) -> None:
        """Initialize the regressor and set default loss and metric."""
        super(TabNetRegressor, self).__post_init__()
        self._task: str = "regression"
        # self._default_loss: Any = torch.nn.functional.mse_loss
        self._default_loss: Any = partial(
            torch.nn.functional.mse_loss,
            reduction="none",
        )
        self._default_metric: str = "mse"

    def prepare_target(self, y: np.ndarray) -> np.ndarray:
        """Return the input as target for regression.

        Parameters
        ----------
        y : np.ndarray
            Target array.

        Returns
        -------
        np.ndarray
            Same as input.

        """
        return y

    def compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        w: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the loss for regression.

        Parameters
        ----------
        y_pred : torch.Tensor
            Network output.
        y_true : torch.Tensor
            True values.
        w : Optional[torch.Tensor]
            Optional sample weights.

        Returns
        -------
        torch.Tensor
            Loss value.

        """
        loss = self.loss_fn(
            y_pred,
            y_true,
        )
        if len(loss.shape) > 1:
            loss = torch.mean(loss, dim=1)
        if w is not None:
            loss = loss * w
        return loss.mean()

    def update_fit_params(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: List[Tuple[np.ndarray, np.ndarray]],
        weights: Union[bool, np.ndarray],
    ) -> None:
        """Update fit parameters for regression.

        Parameters
        ----------
        X_train : np.ndarray
            Training data.
        y_train : np.ndarray
            Training targets.
        eval_set : list
            List of evaluation sets.
        weights : bool or np.ndarray
            Sample weights.

        Raises
        ------
        ValueError
            If y_train does not have 2 dimensions.

        """
        if len(y_train.shape) != 2:
            msg: str = (
                "Targets should be 2D : (n_samples, n_regression) "
                + f"but y_train.shape={y_train.shape} given.\n"
                + "Use reshape(-1, 1) for single regression."
            )
            raise ValueError(msg)
        self.output_dim: int = y_train.shape[1]
        self.preds_mapper: None = None

        self.updated_weights: Union[bool, np.ndarray] = weights
        filter_weights(self.updated_weights)

    def predict_func(self, outputs: np.ndarray) -> np.ndarray:
        """Return regression outputs as predictions.

        Parameters
        ----------
        outputs : np.ndarray
            Network outputs.

        Returns
        -------
        np.ndarray
            Regression predictions.

        """
        return outputs

    def stack_batches(
        self,
        list_y_true: List[torch.Tensor],
        list_y_score: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stack batches of true and predicted values for regression.

        Parameters
        ----------
        list_y_true : List[torch.Tensor]
            List of true values for each batch.
        list_y_score : List[torch.Tensor]
            List of predicted values for each batch.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Stacked true values and predicted values.

        """
        y_true: torch.Tensor = torch.vstack(list_y_true)
        y_score: torch.Tensor = torch.vstack(list_y_score)
        return y_true, y_score


# Alias for backward compatibility
MultiTabNetRegressor = TabNetRegressor
