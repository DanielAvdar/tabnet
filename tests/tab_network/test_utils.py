import torch

from pytorch_tabnet.tab_network.utils_funcs import initialize_glu, initialize_non_glu


def test_utils_funcs():
    input_dim = 10
    output_dim = 5

    # Test initialize_non_glu
    linear_module = torch.nn.Linear(input_dim, output_dim)
    initialize_non_glu(linear_module, input_dim, output_dim)

    # Test initialize_glu
    glu_module = torch.nn.Linear(input_dim, 2 * output_dim)  # GLU has 2*output_dim
    initialize_glu(glu_module, input_dim, output_dim)
