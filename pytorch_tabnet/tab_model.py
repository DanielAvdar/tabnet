import torch
import numpy as np
from scipy.special import softmax
from pytorch_tabnet.utils import SparsePredictDataset, PredictDataset, filter_weights
from pytorch_tabnet.abstract_model import TabModel
from pytorch_tabnet.multiclass_utils import infer_output_dim, check_output_dim
from torch.utils.data import DataLoader
import scipy
from typing import Union, Dict, Any, Tuple, List


class TabNetClassifier(TabModel):
    def __post_init__(self) -> None:
        super(TabNetClassifier, self).__post_init__()
        self._task: str = "classification"
        self._default_loss: Any = torch.nn.functional.cross_entropy
        self._default_metric: str = "accuracy"

    def weight_updater(
        self, weights: Union[bool, Dict[str, Any]]
    ) -> Union[bool, Dict[str, Any]]:
        """
        Updates weights dictionary according to target_mapper.

        Parameters
        ----------
        weights : bool or dict
            Given weights for balancing training.

        Returns
        -------
        bool or dict
            Same bool if weights are bool, updated dict otherwise.

        """
        if isinstance(weights, int):
            return weights
        elif isinstance(weights, dict):
            return {self.target_mapper[key]: value for key, value in weights.items()}
        else:
            return weights

    def prepare_target(self, y: np.ndarray) -> np.ndarray:
        return np.vectorize(self.target_mapper.get)(y)

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(y_pred, y_true.long())

    def update_fit_params(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: List[Tuple[np.ndarray, np.ndarray]],
        weights: Union[bool, Dict[str, Any]],
    ) -> None:
        output_dim: int
        train_labels: List[Any]
        output_dim, train_labels = infer_output_dim(y_train)
        for X, y in eval_set:
            check_output_dim(train_labels, y)
        self.output_dim: int = output_dim
        self._default_metric: str = "auc" if self.output_dim == 2 else "accuracy"
        self.classes_: List[Any] = train_labels
        self.target_mapper: Dict[Any, int] = {
            class_label: index for index, class_label in enumerate(self.classes_)
        }
        self.preds_mapper: Dict[str, Any] = {
            str(index): class_label for index, class_label in enumerate(self.classes_)
        }
        self.updated_weights: Union[bool, Dict[str, Any]] = self.weight_updater(weights)

    def stack_batches(
        self,
        list_y_true: List[np.ndarray],
        list_y_score: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_true: np.ndarray = np.hstack(list_y_true)
        y_score: np.ndarray = np.vstack(list_y_score)
        y_score: np.ndarray = softmax(y_score, axis=1)
        return y_true, y_score

    def predict_func(self, outputs: np.ndarray) -> np.ndarray:
        outputs: np.ndarray = np.argmax(outputs, axis=1)
        return np.vectorize(self.preds_mapper.get)(outputs.astype(str))

    def predict_proba(self, X: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Make predictions for classification on a batch (valid)

        Parameters
        ----------
        X : a :tensor: `torch.Tensor` or matrix: `scipy.sparse.csr_matrix`
            Input data

        Returns
        -------
        res : np.ndarray

        """
        self.network.eval()

        if scipy.sparse.issparse(X):
            dataloader: DataLoader = DataLoader(
                SparsePredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            dataloader: DataLoader = DataLoader(
                PredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )

        results: List[np.ndarray] = []
        for batch_nb, data in enumerate(dataloader):
            data: torch.Tensor = data.to(self.device).float()

            output: torch.Tensor
            _M_loss: torch.Tensor
            output, _M_loss = self.network(data)
            predictions: np.ndarray = (
                torch.nn.Softmax(dim=1)(output).cpu().detach().numpy()
            )
            results.append(predictions)
        res: np.ndarray = np.vstack(results)
        return res


class TabNetRegressor(TabModel):
    def __post_init__(self) -> None:
        super(TabNetRegressor, self).__post_init__()
        self._task: str = "regression"
        self._default_loss: Any = torch.nn.functional.mse_loss
        self._default_metric: str = "mse"

    def prepare_target(self, y: np.ndarray) -> np.ndarray:
        return y

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(y_pred, y_true)

    def update_fit_params(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: List[Tuple[np.ndarray, np.ndarray]],
        weights: Union[bool, np.ndarray],
    ) -> None:
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
        return outputs

    def stack_batches(
        self,
        list_y_true: List[np.ndarray],
        list_y_score: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_true: np.ndarray = np.vstack(list_y_true)
        y_score: np.ndarray = np.vstack(list_y_score)
        return y_true, y_score
