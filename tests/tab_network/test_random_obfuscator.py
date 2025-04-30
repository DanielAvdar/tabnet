import torch

from pytorch_tabnet.tab_network.random_obfuscator import RandomObfuscator


def test_random_obfuscator():
    bs = 32
    input_dim = 16
    pretraining_ratio = 0.2
    group_matrix = torch.randint(0, 2, size=(5, input_dim)).float()
    obfuscator = RandomObfuscator(pretraining_ratio, group_matrix)
    x = torch.rand((bs, input_dim))

    masked_x, obfuscated_groups, obfuscated_vars = obfuscator.forward(x)
    assert masked_x.shape == x.shape
    assert obfuscated_groups.shape == (bs, group_matrix.shape[0])
    assert obfuscated_vars.shape == x.shape
