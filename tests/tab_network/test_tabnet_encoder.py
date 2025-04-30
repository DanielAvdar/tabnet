import torch

from pytorch_tabnet.tab_network.tabnet_encoder import TabNetEncoder


def test_tabnet_encoder():
    input_dim = 16
    output_dim = 8
    n_d = 8
    n_a = 8
    n_steps = 3
    gamma = 1.3
    n_independent = 2
    n_shared = 2

    encoder = TabNetEncoder(
        input_dim=input_dim,
        output_dim=output_dim,
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
        n_independent=n_independent,
        n_shared=n_shared,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
    )

    batch_size = 10
    x = torch.rand((batch_size, input_dim))

    steps_output, M_loss = encoder.forward(x)

    assert len(steps_output) == n_steps
    assert steps_output[0].shape == (batch_size, n_d)
    assert isinstance(M_loss, torch.Tensor)
    assert M_loss.numel() == 1  # Scalar loss


def test_tabnet_encoder_forward_masks():
    """Test the forward_masks method of TabNetEncoder."""
    input_dim = 16
    output_dim = 8
    n_d = 8
    n_a = 8
    n_steps = 3
    gamma = 1.3

    encoder = TabNetEncoder(
        input_dim=input_dim,
        output_dim=output_dim,
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
    )

    batch_size = 10
    x = torch.rand((batch_size, input_dim))

    # Test forward_masks method - this returns a tuple of (explanation_mask, masks_dict)
    explanation_mask, masks_dict = encoder.forward_masks(x)

    # Check the return structure and shapes
    assert explanation_mask.shape == (batch_size, input_dim)
    assert isinstance(masks_dict, dict)
    assert len(masks_dict) == n_steps
    for step in range(n_steps):
        assert step in masks_dict
        assert masks_dict[step].shape == (batch_size, input_dim)


def test_tabnet_encoder_group_matrix():
    """Test TabNetEncoder with a group matrix."""
    input_dim = 16
    output_dim = 8
    n_d = 8
    n_a = 8
    n_steps = 3
    n_groups = 4

    # Create a group matrix
    group_matrix = torch.randint(0, 2, size=(n_groups, input_dim)).float()

    encoder = TabNetEncoder(
        input_dim=input_dim, output_dim=output_dim, n_d=n_d, n_a=n_a, n_steps=n_steps, group_attention_matrix=group_matrix
    )

    batch_size = 10
    x = torch.rand((batch_size, input_dim))

    steps_output, M_loss = encoder.forward(x)

    assert len(steps_output) == n_steps
    assert steps_output[0].shape == (batch_size, n_d)
    assert isinstance(M_loss, torch.Tensor)
