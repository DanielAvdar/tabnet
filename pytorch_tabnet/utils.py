"""Utility functions for TabNet package."""

import json
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy
import torch
from sklearn.utils import check_array


def create_explain_matrix(
    input_dim: int,
    cat_emb_dim: Union[int, List[int]],
    cat_idxs: List[int],
    post_embed_dim: int,
) -> scipy.sparse.csc_matrix:
    """Create a reducing matrix for summing importances from embeddings to initial indices.

    In order to rapidly sum importances from same embeddings to the initial index.

    Parameters
    ----------
    input_dim : int
        Initial input dim
    cat_emb_dim : int or list of int
        if int : size of embedding for all categorical feature
        if list of int : size of embedding for each categorical feature
    cat_idxs : list of int
        Initial position of categorical features
    post_embed_dim : int
        Post embedding inputs dimension

    Returns
    -------
    reducing_matrix : np.array
        Matrix of dim (post_embed_dim, input_dim) to perform reduce

    """
    if isinstance(cat_emb_dim, int):
        all_emb_impact = [cat_emb_dim - 1] * len(cat_idxs)
    else:
        all_emb_impact = [emb_dim - 1 for emb_dim in cat_emb_dim]

    acc_emb = 0
    nb_emb = 0
    indices_trick = []
    for i in range(input_dim):
        if i not in cat_idxs:
            indices_trick.append([i + acc_emb])
        else:
            indices_trick.append(
                range(i + acc_emb, i + acc_emb + all_emb_impact[nb_emb] + 1)  # type: ignore
            )
            acc_emb += all_emb_impact[nb_emb]
            nb_emb += 1

    reducing_matrix = np.zeros((post_embed_dim, input_dim))
    for i, cols in enumerate(indices_trick):
        reducing_matrix[cols, i] = 1

    return scipy.sparse.csc_matrix(reducing_matrix)


def create_group_matrix(list_groups: List[List[int]], input_dim: int) -> torch.Tensor:
    """Create the group matrix corresponding to the given list_groups.

    Parameters
    ----------
    list_groups : list of list of int
        Each element is a list representing features in the same group.
        One feature should appear in maximum one group.
        Feature that don't get assigned a group will be in their own group of one feature.
    input_dim : int
        Number of features in the initial dataset.

    Returns
    -------
    group_matrix : torch.Tensor
        A matrix of size (n_groups, input_dim) where m_ij represents the importance of feature j in group i.
        The rows must sum to 1 as each group is equally important a priori.

    """
    check_list_groups(list_groups, input_dim)

    if len(list_groups) == 0:
        group_matrix = torch.eye(input_dim)
        return group_matrix
    else:
        n_groups = input_dim - int(np.sum([len(gp) - 1 for gp in list_groups]))
        group_matrix = torch.zeros((n_groups, input_dim))

        remaining_features = [feat_idx for feat_idx in range(input_dim)]

        current_group_idx = 0
        for group in list_groups:
            group_size = len(group)
            for elem_idx in group:
                # add importrance of element in group matrix and corresponding group
                group_matrix[current_group_idx, elem_idx] = 1 / group_size
                # remove features from list of features
                remaining_features.remove(elem_idx)
            # move to next group
            current_group_idx += 1
        # features not mentionned in list_groups get assigned their own group of singleton
        for remaining_feat_idx in remaining_features:
            group_matrix[current_group_idx, remaining_feat_idx] = 1
            current_group_idx += 1
        return group_matrix


def check_list_groups(list_groups: List[List[int]], input_dim: int) -> None:
    """Check that list_groups is valid for group matrix construction.

    - Is a list of list
    - Does not contain the same feature in different groups
    - Does not contain unknown features (>= input_dim)
    - Does not contain empty groups.

    Parameters
    ----------
    list_groups : list of list of int
        Each element is a list representing features in the same group.
        One feature should appear in maximum one group.
        Feature that don't get assigned a group will be in their own group of one feature.
    input_dim : int
        Number of features in the initial dataset

    """
    assert isinstance(list_groups, list), "list_groups must be a list of list."

    if len(list_groups) == 0:
        return
    else:
        for group_pos, group in enumerate(list_groups):
            msg = f"Groups must be given as a list of list, but found {group} in position {group_pos}."  # noqa
            assert isinstance(group, list), msg
            assert len(group) > 0, "Empty groups are forbidding please remove empty groups []"

    n_elements_in_groups = np.sum([len(group) for group in list_groups])
    flat_list = []
    for group in list_groups:
        flat_list.extend(group)
    unique_elements = np.unique(flat_list)
    n_unique_elements_in_groups = len(unique_elements)
    msg = "One feature can only appear in one group, please check your grouped_features."
    assert n_unique_elements_in_groups == n_elements_in_groups, msg

    highest_feat = np.max(unique_elements)
    assert highest_feat < input_dim, f"Number of features is {input_dim} but one group contains {highest_feat}."  # noqa
    return


def filter_weights(weights: Union[int, List, np.ndarray]) -> None:
    """Ensure weights are in correct format for regression and multitask TabNet.

    Parameters
    ----------
    weights : int, dict or list
        Initial weights parameters given by user

    Raises
    ------
    ValueError
        If weights are not in the correct format for regression, multitask, or pretraining.

    """
    err_msg = """Please provide a list or np.array of weights for """
    err_msg += """regression, multitask or pretraining: """
    if isinstance(weights, int):
        if weights == 1:
            raise ValueError(err_msg + "1 given.")
    if isinstance(weights, dict):
        raise ValueError(err_msg + "Dict given.")
    return


def validate_eval_set(
    eval_set: List[Tuple[np.ndarray, np.ndarray]],
    eval_name: Optional[List[str]],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[List[str], List[Tuple[np.ndarray, np.ndarray]]]:
    """Check if the shapes of eval_set are compatible with (X_train, y_train).

    Parameters
    ----------
    eval_set : list of tuple
        List of eval tuple set (X, y).
        The last one is used for early stopping
    eval_name : list of str
        List of eval set names.
    X_train : np.ndarray
        Train owned products
    y_train : np.array
        Train targeted products

    Returns
    -------
    eval_names : list of str
        Validated list of eval_names.
    eval_set : list of tuple
        Validated list of eval_set.

    """
    eval_name = eval_name or [f"val_{i}" for i in range(len(eval_set))]

    assert len(eval_set) == len(eval_name), "eval_set and eval_name have not the same length"
    if len(eval_set) > 0:
        assert all(len(elem) == 2 for elem in eval_set), "Each tuple of eval_set need to have two elements"
    for name, (X, y) in zip(eval_name, eval_set, strict=False):
        check_input(X)
        msg = f"Dimension mismatch between X_{name} " + f"{X.shape} and X_train {X_train.shape}"
        assert len(X.shape) == len(X_train.shape), msg

        msg = f"Dimension mismatch between y_{name} " + f"{y.shape} and y_train {y_train.shape}"
        assert len(y.shape) == len(y_train.shape), msg

        msg = f"Number of columns is different between X_{name} " + f"({X.shape[1]}) and X_train ({X_train.shape[1]})"
        assert X.shape[1] == X_train.shape[1], msg

        if len(y_train.shape) == 2:
            msg = f"Number of columns is different between y_{name} " + f"({y.shape[1]}) and y_train ({y_train.shape[1]})"
            assert y.shape[1] == y_train.shape[1], msg
        msg = f"You need the same number of rows between X_{name} " + f"({X.shape[0]}) and y_{name} ({y.shape[0]})"
        assert X.shape[0] == y.shape[0], msg

    return eval_name, eval_set


def define_device(device_name: str) -> str:
    """Define the device to use during training and inference.

    If auto it will detect automatically whether to use cuda or cpu.

    Parameters
    ----------
    device_name : str
        Either "auto", "cpu" or "cuda"

    Returns
    -------
    str
        Either "cpu" or "cuda"

    """
    if device_name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    elif device_name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    else:
        return device_name


class ComplexEncoder(json.JSONEncoder):
    """Custom JSON encoder for complex numbers and numpy data types."""

    def default(self, obj: object) -> object:
        """Convert numpy objects to lists for JSON serialization.

        Parameters
        ----------
        obj : object
            The object to encode.

        Returns
        -------
        object
            A JSON-serializable object (list, dict, etc.) or calls the base class default method.

        """
        if isinstance(obj, (np.generic, np.ndarray)):
            return obj.tolist()
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def check_input(X: np.ndarray) -> None:
    """Raise a clear error if X is a pandas dataframe.

    Also check array according to scikit rules.
    """
    if isinstance(X, (pd.DataFrame, pd.Series)):
        err_message = "Pandas DataFrame are not supported: apply X.values when calling fit"
        raise TypeError(err_message)
    check_array(X, accept_sparse=True)


def check_warm_start(warm_start: bool, from_unsupervised: Optional[bool]) -> None:
    """Warn about ambiguous usage of warm_start and from_unsupervised parameters."""
    if warm_start and from_unsupervised is not None:
        warn_msg = "warm_start=True and from_unsupervised != None: "
        warn_msg = "warm_start will be ignore, training will start from unsupervised weights"
        warnings.warn(warn_msg, stacklevel=2)
    return


def check_embedding_parameters(
    cat_dims: List[int], cat_idxs: List[int], cat_emb_dim: Union[int, List[int]]
) -> Tuple[List[int], List[int], List[int]]:
    """Check parameters related to embeddings and rearrange them in a unique manner."""
    if (cat_dims == []) ^ (cat_idxs == []):
        if cat_dims == []:
            msg = "If cat_idxs is non-empty, cat_dims must be defined as a list of same length."
        else:
            msg = "If cat_dims is non-empty, cat_idxs must be defined as a list of same length."
        raise ValueError(msg)
    elif len(cat_dims) != len(cat_idxs):
        msg = "The lists cat_dims and cat_idxs must have the same length."
        raise ValueError(msg)

    if isinstance(cat_emb_dim, int):
        cat_emb_dims = [cat_emb_dim] * len(cat_idxs)
    else:
        cat_emb_dims = cat_emb_dim

    # check that all embeddings are provided
    if len(cat_emb_dims) != len(cat_dims):
        msg = f"""cat_emb_dim and cat_dims must be lists of same length, got {len(cat_emb_dims)}
                    and {len(cat_dims)}"""
        raise ValueError(msg)

    # Rearrange to get reproducible seeds with different ordering
    if len(cat_idxs) > 0:
        sorted_idxs = np.argsort(cat_idxs)
        cat_dims = [cat_dims[i] for i in sorted_idxs]
        cat_emb_dims = [cat_emb_dims[i] for i in sorted_idxs]

    return cat_dims, cat_idxs, cat_emb_dims
