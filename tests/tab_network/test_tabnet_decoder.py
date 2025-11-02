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


def test_tabnet_decoder_no_shared():
    """Test TabNetDecoder with n_shared=0, which means no shared feature transformer."""
    input_dim = 16
    n_d = 8
    n_steps = 3
    n_independent = 2

    # Create a decoder with n_shared=0
    decoder = TabNetDecoder(
        input_dim=input_dim, n_d=n_d, n_steps=n_steps, n_independent=n_independent, n_shared=0, virtual_batch_size=128, momentum=0.02
    )

    batch_size = 10
    steps_output = [torch.rand((batch_size, n_d)) for _ in range(n_steps)]

    out = decoder.forward(steps_output)

    assert out.shape == (batch_size, input_dim)


def test_tabnet_decoder_device_movement():
    """Test that TabNetDecoder moves to the correct device with the model."""
    input_dim = 16
    n_d = 8
    n_steps = 3

    decoder = TabNetDecoder(
        input_dim=input_dim,
        n_d=n_d,
        n_steps=n_steps,
        n_independent=2,
        n_shared=2,
        virtual_batch_size=2,
        momentum=0.02,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    decoder = decoder.to(device)

    batch_size = 2
    steps_output = [torch.rand((batch_size, n_d)).to(device) for _ in range(n_steps)]

    out = decoder.forward(steps_output)
    assert out.shape == (batch_size, input_dim)
