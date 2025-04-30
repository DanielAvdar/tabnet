import torch

from pytorch_tabnet.tab_network.feat_transformer import FeatTransformer


def test_feat_transformer():
    input_dim = 10
    output_dim = 8
    shared_layers = torch.nn.ModuleList([torch.nn.Linear(10, 16)])

    transformer = FeatTransformer(
        input_dim,
        output_dim,
        shared_layers,
        n_glu_independent=2,
        virtual_batch_size=128,
        momentum=0.02,
    )

    bs = 3
    input_data = torch.rand((bs, input_dim))

    output = transformer.forward(input_data)
    assert output.shape == (bs, output_dim)


def test_feat_transformer_no_shared_layers():
    """Test FeatTransformer with no shared layers."""
    input_dim = 10
    output_dim = 8

    # Test with shared_layers=None
    transformer = FeatTransformer(
        input_dim,
        output_dim,
        shared_layers=None,
        n_glu_independent=2,
        virtual_batch_size=128,
        momentum=0.02,
    )

    bs = 3
    input_data = torch.rand((bs, input_dim))
    output = transformer.forward(input_data)
    assert output.shape == (bs, output_dim)


def test_feat_transformer_no_independent_layers():
    """Test FeatTransformer with no independent layers."""
    input_dim = 10
    output_dim = 8
    shared_layers = torch.nn.ModuleList([torch.nn.Linear(10, 16)])

    # Test with n_glu_independent=0
    transformer = FeatTransformer(
        input_dim,
        output_dim,
        shared_layers=shared_layers,
        n_glu_independent=0,
        virtual_batch_size=128,
        momentum=0.02,
    )

    bs = 3
    input_data = torch.rand((bs, input_dim))
    output = transformer.forward(input_data)
    assert output.shape == (bs, output_dim)
