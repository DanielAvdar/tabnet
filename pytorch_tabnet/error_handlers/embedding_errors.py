"""Error handling utilities for embedding-related operations in TabNet."""

from typing import List, Tuple, Union

import numpy as np


def check_embedding_parameters(
    cat_dims: List[int], cat_idxs: List[int], cat_emb_dim: Union[int, List[int]]
) -> Tuple[List[int], List[int], List[int]]:
    """Check parameters related to embeddings and rearrange them in a unique manner.

    Parameters
    ----------
    cat_dims : List[int]
        List of dimensions for each categorical feature.
    cat_idxs : List[int]
        List of indices for categorical features.
    cat_emb_dim : Union[int, List[int]]
        Size of embedding for categorical features. If int, same embedding size
        for all categorical features. If list, dimension must match cat_dims.

    Returns
    -------
    Tuple[List[int], List[int], List[int]]
        Sorted cat_dims, cat_idxs, and cat_emb_dims.

    Raises
    ------
    ValueError
        If dimensions of categorical parameters are inconsistent.

    """
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
