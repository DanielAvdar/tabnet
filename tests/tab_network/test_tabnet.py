import torch

from pytorch_tabnet.tab_network.tabnet import TabNet


def test_tabnet():
    input_dim = 16
    output_dim = 8
    n_d = 8
    n_a = 8
    n_steps = 3
    gamma = 1.3
    n_independent = 2
    n_shared = 2
    cat_idxs = [0, 2]
    cat_dims = [3, 4]
    cat_emb_dim = 2

    # Create a group matrix for embeddings
    group_matrix = torch.rand((2, input_dim))

    tabnet = TabNet(
        input_dim=input_dim,
        output_dim=output_dim,
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=cat_emb_dim,
        n_independent=n_independent,
        n_shared=n_shared,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
        group_attention_matrix=group_matrix,
    )

    # Basic test - without grouped features
    batch_size = 10
    x = torch.randint(0, 3, (batch_size, input_dim))

    out, M_loss = tabnet.forward(x)

    assert out.shape == (batch_size, output_dim)
    assert isinstance(M_loss, torch.Tensor)
    assert M_loss.numel() == 1  # Scalar loss

    # Test with grouped_feat - we'll create a grouped version directly
    group_matrix_grouped = torch.rand((3, input_dim))  # 3 groups

    tabnet_with_groups = TabNet(
        input_dim=input_dim,
        output_dim=output_dim,
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=cat_emb_dim,
        n_independent=n_independent,
        n_shared=n_shared,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
        group_attention_matrix=group_matrix_grouped,
    )

    out_with_groups, M_loss_with_groups = tabnet_with_groups.forward(x)

    assert out_with_groups.shape == (batch_size, output_dim)
