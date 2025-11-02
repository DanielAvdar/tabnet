import torch

from pytorch_tabnet.tab_network.gbn import GBN


def test_gbn():
    feature_dim = 16
    batch_size = 10
    virtual_batch_size = 4

    gbn = GBN(feature_dim, momentum=0.1, virtual_batch_size=virtual_batch_size)

    # Test virtual batch mode
    x = torch.rand((batch_size, feature_dim))
    output = gbn(x)
    assert output.shape == x.shape

    # Test with larger virtual_batch_size
    full_batch_gbn = GBN(
        feature_dim,
        momentum=0.1,
        virtual_batch_size=batch_size * 2,  # Ensure it's larger than the batch
    )

    # Use a different tensor
    y = torch.rand((batch_size, feature_dim))
    output_full = full_batch_gbn(y)

    assert output_full.shape == y.shape


def test_gbn_device_movement():
    """Test that GBN moves to the correct device with the model."""
    feature_dim = 16
    batch_size = 2
    virtual_batch_size = 2

    gbn = GBN(feature_dim, momentum=0.1, virtual_batch_size=virtual_batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    gbn = gbn.to(device)

    x = torch.rand((batch_size, feature_dim)).to(device)
    output = gbn(x)
    assert output.shape == x.shape
