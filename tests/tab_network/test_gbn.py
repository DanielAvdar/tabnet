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
