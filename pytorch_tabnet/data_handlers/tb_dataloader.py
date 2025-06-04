# Empty file for TBDataLoader class
import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union

import torch

from .data_types import tn_type
from .predict_dataset import PredictDataset
from .sparse_predict_dataset import SparsePredictDataset
from .torch_dataset import TorchDataset


@dataclass
class TBDataLoader:
    name: str
    dataset: Union[PredictDataset, TorchDataset]
    batch_size: int
    weights: Optional[torch.Tensor] = None
    pre_training: bool = False
    drop_last: bool = False
    pin_memory: bool = False
    predict: bool = False
    all_at_once: bool = False

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, tn_type, tn_type]]:
        if self.all_at_once:
            if self.dataset.y is None:
                if not self.pre_training:
                    raise ValueError("Dataset has no labels (y), but pre_training is False.")
                yield self.dataset.x, None, None
            elif self.pre_training or isinstance(self.dataset, PredictDataset) or isinstance(self.dataset, SparsePredictDataset): # self.dataset.y is not None here
                yield self.dataset.x, None, None # Pretraining still yields None for y
            else: # not pre_training and self.dataset.y is not None
                yield self.dataset.x, self.dataset.y, None
            return
        ds_len = len(self.dataset)
        perm = None
        if not self.predict:
            perm = torch.randperm(ds_len, pin_memory=self.pin_memory)
        batched_starts = [i for i in range(0, ds_len, self.batch_size)]
        batched_starts += [0] if len(batched_starts) == 0 else []
        for start in batched_starts[: len(self)]:
            if self.predict:
                yield self.make_predict_batch(ds_len, start)
            else:
                yield self.make_train_batch(ds_len, perm, start)

    def make_predict_batch(self, ds_len: int, start: int) -> Tuple[torch.Tensor, tn_type, None]:
        end_at = start + self.batch_size
        if end_at > ds_len:
            end_at = ds_len
        x, y, w = None, None, None
        x = self.dataset.x[start:end_at]
        if self.pre_training:
            y = None
        elif isinstance(self.dataset, PredictDataset) or isinstance(self.dataset, SparsePredictDataset):
            # These datasets might not have 'y' or it might behave differently
            # For now, assume they might not have y, similar to pre_training
            y = None
        else:
            if self.dataset.y is None:
                raise ValueError("Dataset has no labels (y) for supervised learning/prediction.")
            y = self.dataset.y[start:end_at]
        w = None if self.weights is None else self.weights[start:end_at]

        return x, y, w

    def make_train_batch(self, ds_len: int, perm: Optional[torch.Tensor], start: int) -> Tuple[torch.Tensor, tn_type, tn_type]:
        end_at = start + self.batch_size
        left_over = None
        if end_at > ds_len:
            left_over = end_at - ds_len
            end_at = ds_len
        indexes = perm[start:end_at]
        x, y, w = None, None, None
        if self.pre_training:
            x = self.dataset.x[indexes]
            # y remains None for pre-training
        else: # not self.pre_training
            if getattr(self.dataset, 'y', None) is None:
                raise ValueError("Dataset has no labels (y) for supervised training.")
            x, y = self.dataset.x[indexes], self.dataset.y[indexes]
        w = self.get_weights(indexes)

        if left_over is not None:
            lo_indexes = perm[:left_over]

            if self.pre_training:
                x = torch.cat((x, self.dataset.x[lo_indexes]))
                # y remains None for pre-training
                # For w, if pre_training, we need to handle if self.weights is None then w is None
                # or if w was None initially (e.g. first batch part was also pretraining)
                current_w = self.get_weights(lo_indexes)
                if w is None and current_w is not None: # w was None, but this part has weights
                     w = current_w
                elif w is not None and current_w is not None: # w existed, this part has weights, so cat
                     w = torch.cat((w, current_w))
                # if current_w is None, w remains as it was (either None or previous weights)

            else: # not self.pre_training
                # y was already checked for None above, so self.dataset.y exists
                x = torch.cat((x, self.dataset.x[lo_indexes]))
                y = torch.cat((y, self.dataset.y[lo_indexes]))
                # For w, if not pre_training, self.weights might be None or w might be None
                current_w = self.get_weights(lo_indexes)
                if w is None and current_w is not None:
                    w = current_w
                elif w is not None and current_w is not None: # Original logic for not pre_training
                    w = torch.cat((w, current_w))
                # if current_w is None, w remains as it was

        return x, y, w

    def __len__(self) -> int:
        res = math.ceil(len(self.dataset) / self.batch_size)
        need_to_drop_last = self.drop_last and not self.predict
        need_to_drop_last = need_to_drop_last and (res > 1)
        res -= need_to_drop_last
        return res

    def get_weights(self, i: torch.Tensor = None) -> Union[torch.Tensor, None]:
        if self.weights is None:
            return None
        if i is None:
            return self.weights
        return self.weights[i]
