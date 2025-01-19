from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader

from pytorch_tabnet.abstract_model import TabModel
from pytorch_tabnet.data_handlers import PredictDataset, SparsePredictDataset
from pytorch_tabnet.multiclass_utils import check_output_dim, infer_multitask_output
from pytorch_tabnet.utils import filter_weights


@dataclass
class TabNetMultiTaskClassifier(TabModel):
    output_dim: List[int] = None

    def __post_init__(self) -> None:
        super(TabNetMultiTaskClassifier, self).__post_init__()
        self._task = "classification"
        self._default_loss = torch.nn.functional.cross_entropy
        self._default_metric = "logloss"

    def prepare_target(self, y: np.ndarray) -> np.ndarray:
        y_mapped = y.copy()
        for task_idx in range(y.shape[1]):
            task_mapper = self.target_mapper[task_idx]
            y_mapped[:, task_idx] = np.vectorize(task_mapper.get)(y[:, task_idx])
        return y_mapped

    def compute_loss(self, y_pred: List[torch.Tensor], y_true: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss according to network output and targets

        Parameters
        ----------
        y_pred : list of tensors
            Output of network
        y_true : LongTensor
            Targets label encoded

        Returns
        -------
        loss : torch.Tensor
            output of loss function(s)

        """
        loss = 0.0
        y_true = y_true.long()
        if isinstance(self.loss_fn, list):
            # if you specify a different loss for each task
            for task_loss, task_output, task_id in zip(self.loss_fn, y_pred, range(len(self.loss_fn)), strict=False):
                loss += task_loss(task_output, y_true[:, task_id])
        else:
            # same loss function is applied to all tasks
            for task_id, task_output in enumerate(y_pred):
                loss += self.loss_fn(task_output, y_true[:, task_id])

        loss /= len(y_pred)
        return loss

    def stack_batches(
        self,
        list_y_true: List[torch.Tensor],
        list_y_score: List[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        y_true = torch.vstack(list_y_true)
        y_score = []
        for i in range(len(self.output_dim)):
            score = torch.vstack([x[i] for x in list_y_score])
            score = torch.nn.Softmax(dim=1)(score)
            y_score.append(score)
        return y_true, y_score

    def update_fit_params(
        self,
        X_train: Union[np.ndarray, scipy.sparse.csr_matrix],
        y_train: np.ndarray,
        eval_set: List[Tuple[Union[np.ndarray, scipy.sparse.csr_matrix], np.ndarray]],
        weights: Union[np.ndarray, None],
    ) -> None:
        output_dim, train_labels = infer_multitask_output(y_train)
        for _, y in eval_set:
            for task_idx in range(y.shape[1]):
                check_output_dim(train_labels[task_idx], y[:, task_idx])
        self.output_dim = output_dim
        self.classes_ = train_labels
        self.target_mapper = [{class_label: index for index, class_label in enumerate(classes)} for classes in self.classes_]
        self.preds_mapper = [{str(index): str(class_label) for index, class_label in enumerate(classes)} for classes in self.classes_]
        # self.updated_weights = weights
        filter_weights(
            weights=weights,
        )

    def predict(self, X: Union[torch.Tensor, np.ndarray, scipy.sparse.csr_matrix]) -> List[np.ndarray]:
        """
        Make predictions on a batch (valid)

        Parameters
        ----------
        X : a :tensor: `torch.Tensor` or matrix: `scipy.sparse.csr_matrix`
            Input data

        Returns
        -------
        results : np.array
            Predictions of the most probable class
        """
        self.network.eval()

        if scipy.sparse.issparse(X):
            dataloader = DataLoader(
                SparsePredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            dataloader = DataLoader(
                PredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )

        results: dict = {}
        with torch.no_grad():
            for data in dataloader:
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

    def predict_proba(self, X: Union[torch.Tensor, np.ndarray, scipy.sparse.csr_matrix]) -> List[np.ndarray]:
        """
        Make predictions for classification on a batch (valid)

        Parameters
        ----------
        X : a :tensor: `torch.Tensor` or matrix: `scipy.sparse.csr_matrix`
            Input data

        Returns
        -------
        res : list of np.ndarray

        """
        self.network.eval()

        if scipy.sparse.issparse(X):
            dataloader = DataLoader(
                SparsePredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            dataloader = DataLoader(
                PredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )

        results: dict = {}
        for data in dataloader:
            data = data.to(self.device).float()
            output, _ = self.network(data)
            predictions = [torch.nn.Softmax(dim=1)(task_output).cpu().detach().numpy() for task_output in output]
            for task_idx in range(len(self.output_dim)):
                results[task_idx] = results.get(task_idx, []) + [predictions[task_idx]]
        res = [np.vstack(task_res) for task_res in results.values()]
        return res
