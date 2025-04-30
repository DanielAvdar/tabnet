"""Device utility functions for TabNet."""

import torch


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
