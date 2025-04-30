import torch

from pytorch_tabnet.tab_network.tabnet_noembeddings import TabNetNoEmbeddings


def test_tabnet_no_embeddings():
    input_dim = 16
    output_dim = 8
    n_d = 8
    n_a = 8
    n_steps = 3
    gamma = 1.3
    n_independent = 2
    n_shared = 2

    # Create a group matrix for testing
    group_matrix = torch.rand((2, input_dim))

    tabnet_no_emb = TabNetNoEmbeddings(
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

    # Basic test
    batch_size = 10
    x = torch.rand((batch_size, input_dim))

    out, M_loss = tabnet_no_emb.forward(x)

    assert out.shape == (batch_size, output_dim)
    assert isinstance(M_loss, torch.Tensor)
    assert M_loss.numel() == 1  # Scalar loss

    # Test with group_attention_matrix
    tabnet_no_emb_with_groups = TabNetNoEmbeddings(
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
        group_attention_matrix=group_matrix,
    )

    out_with_groups, M_loss_with_groups = tabnet_no_emb_with_groups.forward(x)

    assert out_with_groups.shape == (batch_size, output_dim)


def test_tabnet_no_embeddings_forward_masks():
    """Test the forward_masks method of TabNetNoEmbeddings."""
    input_dim = 16
    output_dim = 8
    n_d = 8
    n_a = 8
    n_steps = 3

    tabnet_no_emb = TabNetNoEmbeddings(
        input_dim=input_dim,
        output_dim=output_dim,
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
    )

    batch_size = 10
    x = torch.rand((batch_size, input_dim))

    # Test forward_masks method
    global_mask, masks_dict = tabnet_no_emb.forward_masks(x)

    assert global_mask.shape == (batch_size, input_dim)
    assert isinstance(masks_dict, dict)
    assert len(masks_dict) == n_steps


def test_tabnet_no_embeddings_multi_output():
    """Test TabNetNoEmbeddings with multiple output dimensions."""
    input_dim = 16
    # Define multiple output dimensions for multi-task
    output_dim = [8, 6, 4]
    n_d = 8
    n_a = 8
    n_steps = 3

    tabnet_no_emb = TabNetNoEmbeddings(
        input_dim=input_dim,
        output_dim=output_dim,
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
    )

    batch_size = 10
    x = torch.rand((batch_size, input_dim))

    outputs, M_loss = tabnet_no_emb.forward(x)

    # Check that we get multiple outputs matching the specified dimensions
    assert isinstance(outputs, list)
    assert len(outputs) == len(output_dim)
    for i, dim in enumerate(output_dim):
        assert outputs[i].shape == (batch_size, dim)
