from typing import Dict, Tuple, Union

import numpy as np
import torch

from pytorch_tabnet.data_handlers import PredictDataset, TBDataLoader


def explain_v1(
    X: Union[np.ndarray, torch.Tensor],
    batch_size: int,
    device: torch.device,
    network: torch.nn.Module,
    normalize: bool,
    reducing_matrix: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return local feature importance using numpy operations.

    Parameters
    ----------
    X : Union[np.ndarray, torch.Tensor]
        Input data
    batch_size : int
        Batch size for DataLoader
    device : torch.device
        Device to run computations on
    network : torch.nn.Module
        Network to compute masks
    normalize : bool
        Whether to normalize importance so that contributions sum to 1
    reducing_matrix : np.ndarray
        Matrix for dimensionality reduction in numpy format

    Returns
    -------
    Tuple[np.ndarray, Dict[str, np.ndarray]]
        Tuple containing:
        - Feature importance matrix (n_samples, n_features)
        - Dictionary of masks
    """
    dataloader = TBDataLoader(
        name="predict",
        dataset=PredictDataset(X),
        batch_size=batch_size,
        predict=True,
    )

    res_explain = []
    res_masks = None

    # Set random seed to ensure consistent results
    torch.manual_seed(0)

    with torch.no_grad():
        for batch_nb, (data, _, _) in enumerate(dataloader):  # type: ignore
            data = data.to(device, non_blocking=True).float()  # type: ignore

            M_explain, masks = network.forward_masks(data)
            # Convert to numpy and ensure same dtype as reducing_matrix
            M_explain_np = M_explain.cpu().detach().numpy().astype(reducing_matrix.dtype)

            # Process masks
            for key, value in masks.items():
                value_np = value.cpu().detach().numpy().astype(reducing_matrix.dtype)
                # Match scipy's csc_matrix dot behavior
                masks[key] = value_np @ reducing_matrix

            # Match scipy's csc_matrix dot behavior
            original_feat_explain = M_explain_np @ reducing_matrix
            res_explain.append(original_feat_explain)

            if batch_nb == 0:
                res_masks = masks
            else:
                for key, value in masks.items():
                    res_masks[key] = np.vstack([res_masks[key], value])

    res_explain = np.vstack(res_explain)
    if normalize:
        # Match scipy normalization behavior
        sum_axis1 = np.sum(res_explain, axis=1, keepdims=True)
        # Handle division by zero
        sum_axis1[sum_axis1 == 0] = 1
        res_explain = res_explain / sum_axis1

    return res_explain, res_masks
