import pytest
import torch

from pytorch_tabnet.tab_network.attentive_transformer import AttentiveTransformer


@pytest.mark.parametrize("mask_type", ["sparsemax", "entmax"])
def test_attentive_transformer(mask_type):
    input_dim = 8
    group_dim = 5
    group_matrix = torch.randint(0, 2, size=(group_dim, input_dim)).float()

    transformer = AttentiveTransformer(
        input_dim,
        group_dim,
        group_matrix,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type=mask_type,
    )

    bs = 3
    priors = torch.rand((bs, group_dim))
    processed_feat = torch.rand((bs, input_dim))
    output = transformer.forward(priors, processed_feat)
    assert output.shape == (bs, group_dim)


def test_invalid_mask_type():
    """Test that an invalid mask type raises NotImplementedError."""
    input_dim = 8
    group_dim = 5
    group_matrix = torch.randint(0, 2, size=(group_dim, input_dim)).float()

    with pytest.raises(NotImplementedError):
        AttentiveTransformer(
            input_dim,
            group_dim,
            group_matrix,
            virtual_batch_size=128,
            momentum=0.02,
            mask_type="invalid_mask_type",
        )


def test_attentive_transformer_device_movement():
    """Test that AttentiveTransformer moves to the correct device with the model."""
    input_dim = 8
    group_dim = 5
    group_matrix = torch.randint(0, 2, size=(group_dim, input_dim)).float()

    transformer = AttentiveTransformer(
        input_dim,
        group_dim,
        group_matrix,
        virtual_batch_size=2,
        momentum=0.02,
        mask_type="sparsemax",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    transformer = transformer.to(device)

    bs = 2
    priors = torch.rand((bs, group_dim)).to(device)
    processed_feat = torch.rand((bs, input_dim)).to(device)

    output = transformer.forward(priors, processed_feat)
    assert output.shape == (bs, group_dim)
