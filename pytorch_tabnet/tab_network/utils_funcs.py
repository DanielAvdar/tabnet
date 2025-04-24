import math

import torch


def initialize_non_glu(module: torch.nn.Module, input_dim: int, output_dim: int) -> None:
    """Initialize a non-GLU (Gated Linear Unit) linear module with Xavier normal initialization."""
    total_dim = input_dim + output_dim
    sq_input_dim = math.sqrt(4 * input_dim)
    gain_value = math.sqrt(total_dim / sq_input_dim)
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return


def initialize_glu(module: torch.nn.Module, input_dim: int, output_dim: int) -> None:
    """Initialize a GLU (Gated Linear Unit) linear module with Xavier normal initialization."""
    gain_value = math.sqrt((input_dim + output_dim) / math.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return
