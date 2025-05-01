"""Multitask learning utilities for TabNet."""

from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np

# import scipy
import torch

# from torch.utils.data import DataLoader
from pytorch_tabnet.abstract_model_sub import TabSupervisedModel
from pytorch_tabnet.data_handlers import PredictDataset, TBDataLoader
from pytorch_tabnet.utils import check_output_dim, filter_weights, infer_multitask_output


@dataclass
class TabNetMultiTaskClassifier(TabSupervisedModel):
    """TabNet model for multitask classification tasks."""

    output_dim: List[int] = None

    def __post_init__(self) -> None:
        """Initialize the multitask classifier and set default loss and metric."""
        super(TabNetMultiTaskClassifier, self).__post_init__()
        self._task = "classification"
        self._default_loss = partial(torch.nn.functional.cross_entropy, reduction="none")
        self._default_metric = "logloss"

    def prepare_target(self, y: np.ndarray) -> np.ndarray:
        """Map targets for each task using the target mappers.

        Parameters
        ----------
        y : np.ndarray
            Target array with shape (n_samples, n_tasks).

        Returns
        -------
        np.ndarray
            Mapped target array.

        """
        y_mapped = y.copy()
        for task_idx in range(y.shape[1]):
            task_mapper = self.target_mapper[task_idx]
            y_mapped[:, task_idx] = np.vectorize(task_mapper.get)(y[:, task_idx])
        return y_mapped

    def compute_loss(
        self,
        y_pred: List[torch.Tensor],
        y_true: torch.Tensor,
        w: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the loss according to network output and targets.

        Parameters
        ----------
        y_pred : list of torch.Tensor
            Output of network for each task.
        y_true : torch.Tensor
            Targets label encoded.
        w : Optional[torch.Tensor]
            Optional sample weights.

        Returns
        -------
        torch.Tensor
            Output of loss function(s).

        """
        loss = 0.0
        y_true = y_true.long()
        if isinstance(self.loss_fn, list):
            # if you specify a different loss for each task
            for task_loss, task_output, task_id in zip(self.loss_fn, y_pred, range(len(self.loss_fn)), strict=False):
                t_loss = task_loss(task_output, y_true[:, task_id])
                if w is not None:
                    t_loss *= w
                loss += t_loss.mean()
        else:
            # same loss function is applied to all tasks
            for task_id, task_output in enumerate(y_pred):
                t_loss = self.loss_fn(task_output, y_true[:, task_id])
                if w is not None:
                    t_loss *= w
                loss += t_loss.mean()

        loss /= len(y_pred)
        return loss

    def stack_batches(
        self,
        list_y_true: List[torch.Tensor],
        list_y_score: List[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Stack batches of true and predicted values for all tasks.

        Parameters
        ----------
        list_y_true : List[torch.Tensor]
            List of true labels for each batch.
        list_y_score : List[List[torch.Tensor]]
            List of predicted scores for each batch and task.

        Returns
        -------
        Tuple[torch.Tensor, List[torch.Tensor]]
            Stacked true labels and list of stacked predicted scores per task.

        """
        y_true = torch.vstack(list_y_true)
        y_score = []
        for i in range(len(self.output_dim)):
            score = torch.vstack([x[i] for x in list_y_score])
            score = torch.nn.Softmax(dim=1)(score)
            y_score.append(score)
        return y_true, y_score

    def update_fit_params(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: List[Tuple[np.ndarray, np.ndarray]],
        weights: Union[np.ndarray, None],
    ) -> None:
        """Update fit parameters for multitask classification.

        Parameters
        ----------
        X_train : np.ndarray or scipy.sparse.csr_matrix
            Training data.
        y_train : np.ndarray
            Training targets.
        eval_set : list
            List of evaluation sets.
        weights : np.ndarray or None
            Sample weights.

        """
        output_dim, train_labels = infer_multitask_output(y_train)
        for _, y in eval_set:
            for task_idx in range(y.shape[1]):
                check_output_dim(train_labels[task_idx], y[:, task_idx])
        self.output_dim = output_dim
        self.classes_ = train_labels
        self.target_mapper = [{class_label: index for index, class_label in enumerate(classes)} for classes in self.classes_]
        self.preds_mapper = [{str(index): str(class_label) for index, class_label in enumerate(classes)} for classes in self.classes_]
        filter_weights(
            weights=weights,
        )

    def predict(self, X: Union[torch.Tensor, np.ndarray]) -> List[np.ndarray]:
        """Predict the most probable class for each task.

        Parameters
        ----------
        X : torch.Tensor, np.ndarray, or scipy.sparse.csr_matrix
            Input data.

        Returns
        -------
        List[np.ndarray]
            List of predictions for each task.

        """
        self.network.eval()

        dataloader = TBDataLoader(
            name="predict",
            dataset=PredictDataset(X),
            batch_size=self.batch_size,
            predict=True,
        )

        results: dict = {}
        with torch.no_grad():
            for data, _, _ in dataloader:  # type: ignore
                data = data.to(self.device).float()
                output, _ = self.network(data)
                predictions = [
                    torch.argmax(torch.nn.Softmax(dim=1)(task_output), dim=1).cpu().detach().numpy().reshape(-1) for task_output in output
                ]

                for task_idx in range(len(self.output_dim)):
                    results[task_idx] = results.get(task_idx, []) + [predictions[task_idx]]
        # stack all task individually
        results_ = [np.hstack(task_res) for task_res in results.values()]
        # map all task individually
        results_ = [np.vectorize(self.preds_mapper[task_idx].get)(task_res.astype(str)) for task_idx, task_res in enumerate(results_)]
        return results_

    def predict_proba(self, X: Union[torch.Tensor, np.ndarray]) -> List[np.ndarray]:
        """Predict class probabilities for each task.

        Parameters
        ----------
        X : torch.Tensor, np.ndarray, or scipy.sparse.csr_matrix
            Input data.

        Returns
        -------
        List[np.ndarray]
            List of probability predictions for each task.

        """
        self.network.eval()

        dataloader = TBDataLoader(
            name="predict",
            dataset=PredictDataset(X),
            batch_size=self.batch_size,
            predict=True,
        )

        results: dict = {}
        for data, _, _ in dataloader:  # type: ignore
            data = data.to(self.device).float()
            output, _ = self.network(data)
            predictions = [torch.nn.Softmax(dim=1)(task_output).cpu().detach().numpy() for task_output in output]
            for task_idx in range(len(self.output_dim)):
                results[task_idx] = results.get(task_idx, []) + [predictions[task_idx]]
        res = [np.vstack(task_res) for task_res in results.values()]
        return res
