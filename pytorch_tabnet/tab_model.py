"""TabNet model class and training logic."""

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# from torch.utils.data import DataLoader
from pytorch_tabnet.abstract_model_sub import TabSupervisedModel
from pytorch_tabnet.data_handlers import PredictDataset, TBDataLoader
from pytorch_tabnet.utils import check_output_dim, filter_weights, infer_output_dim


@dataclass
class TabNetClassifier(TabSupervisedModel):
    """TabNet model for classification tasks."""

    output_dim: int = None
    weight: Any = field(init=False, default=0)

    def __post_init__(self) -> None:
        """Initialize the classifier and set default loss and metric."""
        super(TabNetClassifier, self).__post_init__()
        self._task: str = "classification"
        self._default_loss: Any = partial(
            torch.nn.functional.cross_entropy,
            reduction="none",
        )
        self._default_metric: str = "accuracy"

    def weight_updater(self, weights: Union[bool, Dict[Union[str, int], Any], Any]) -> Union[bool, Dict[Union[str, int], Any]]:
        """Update class weights for training.

        Parameters
        ----------
        weights : bool, dict, or any
            Class weights or indicator.

        Returns
        -------
        bool or dict
            Updated weights.

        """
        if isinstance(weights, int):
            return weights  # type: ignore
        elif isinstance(weights, dict):
            return {self.target_mapper[key]: value for key, value in weights.items()}
        else:
            return weights

    def prepare_target(self, y: np.ndarray) -> np.ndarray:
        """Map targets using the target mapper.

        Parameters
        ----------
        y : np.ndarray
            Target array.

        Returns
        -------
        np.ndarray
            Mapped target array.

        """
        return np.vectorize(self.target_mapper.get)(y)

    def compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        w: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the loss for classification.

        Parameters
        ----------
        y_pred : torch.Tensor
            Network output.
        y_true : torch.Tensor
            True labels.
        w : Optional[torch.Tensor]
            Optional sample weights.

        Returns
        -------
        torch.Tensor
            Loss value.

        """
        class_count = None
        if isinstance(self.weight, int) and self.weight == 1:
            _class_num, class_count = y_true.long().unique(return_counts=True)
            class_count[class_count == 0] = 1

        loss = self.loss_fn(y_pred, y_true.long(), weight=1 / class_count if class_count is not None else None)
        if w is not None:
            loss = loss * w
        return loss.mean()

    def update_fit_params(  # type: ignore[override]
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: List[Tuple[np.ndarray, np.ndarray]],
        weights: Union[bool, Dict[str, Any]],
    ) -> None:
        """Update fit parameters for classification.

        Parameters
        ----------
        X_train : np.ndarray
            Training data.
        y_train : np.ndarray
            Training targets.
        eval_set : list
            List of evaluation sets.
        weights : bool or dict
            Class weights.

        """
        output_dim: int
        train_labels: List[Any]
        output_dim, train_labels = infer_output_dim(y_train)
        for _X, y in eval_set:
            check_output_dim(train_labels, y)
        self.output_dim: int = output_dim
        self._default_metric = "auc" if self.output_dim == 2 else "accuracy"
        self.classes_: List[Any] = train_labels
        self.target_mapper: Dict[Any, int] = {class_label: index for index, class_label in enumerate(self.classes_)}
        self.preds_mapper: Dict[str, Any] = {str(index): class_label for index, class_label in enumerate(self.classes_)}
        # self.updated_weights: Union[bool, Dict[Union[str, int], Any]] = self.weight_updater(weights)

    def stack_batches(
        self,
        list_y_true: List[torch.Tensor],
        list_y_score: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stack batches of true and predicted values.

        Parameters
        ----------
        list_y_true : List[torch.Tensor]
            List of true labels for each batch.
        list_y_score : List[torch.Tensor]
            List of predicted scores for each batch.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Stacked true labels and predicted scores.

        """
        y_true: torch.Tensor = torch.hstack(list_y_true)
        y_score: torch.Tensor = torch.vstack(list_y_score)
        y_score = torch.nn.Softmax(dim=1)(y_score)
        return y_true, y_score

    def predict_func(self, outputs: np.ndarray) -> np.ndarray:
        """Convert network outputs to class predictions.

        Parameters
        ----------
        outputs : np.ndarray
            Network outputs.

        Returns
        -------
        np.ndarray
            Predicted classes.

        """
        outputs = np.argmax(outputs, axis=1)
        return np.vectorize(self.preds_mapper.get)(outputs.astype(str))

    def predict_proba(self, X: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Predict class probabilities for classification.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Probability predictions.

        """
        self.network.eval()

        dataloader = TBDataLoader(
            name="predict",
            dataset=PredictDataset(X),
            batch_size=self.batch_size,
            # shuffle=False,
            predict=True,
        )

        results: List[np.ndarray] = []
        with torch.no_grad():
            for _batch_nb, (data, _, _) in enumerate(dataloader):  # type: ignore
                data = data.to(self.device).float()  # type: ignore

                output: torch.Tensor
                _M_loss: torch.Tensor
                output, _M_loss = self.network(data)
                predictions: np.ndarray = (
                    torch.nn.Softmax(dim=1)(output).cpu().detach().numpy()
                )  # todo: replace with pytorch's torch.vstack
                results.append(predictions)
            res: np.ndarray = np.vstack(results)  # todo: replace with pytorch's torch.vstack
        return res


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


MultiTabNetRegressor = TabNetRegressor
