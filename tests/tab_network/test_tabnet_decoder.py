import torch

from pytorch_tabnet.tab_network.tabnet_decoder import TabNetDecoder


def test_tabnet_decoder():
    input_dim = 16
    n_d = 8
    n_steps = 3
    n_independent = 2
    n_shared = 2

    decoder = TabNetDecoder(
        input_dim=input_dim, n_d=n_d, n_steps=n_steps, n_independent=n_independent, n_shared=n_shared, virtual_batch_size=128, momentum=0.02
    )

    batch_size = 10
    steps_output = [torch.rand((batch_size, n_d)) for _ in range(n_steps)]

    # TabNetDecoder.forward takes a single positional argument, steps_output
    out = decoder.forward(steps_output)

    assert out.shape == (batch_size, input_dim)
