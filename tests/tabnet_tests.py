import torch
import pytest
from pytorch_tabnet.tab_network import (
    AttentiveTransformer,
    FeatTransformer,
    EmbeddingGenerator,
    RandomObfuscator,
)


@pytest.fixture
def embedding_generator():
    input_dim = 10
    cat_dims = [2, 3, 4]
    cat_idxs = [0, 2, 4]
    cat_emb_dims = [1, 2, 3]

    group_matrix = torch.randint(0, 2, size=(3, input_dim)).float()

    return EmbeddingGenerator(input_dim, cat_dims, cat_idxs, cat_emb_dims, group_matrix)


# def test_embedding_generator_forward(embedding_generator):
#     batch_size = 5
#     input_data = torch.randint(0, 10, size=(batch_size, 10))
#     output = embedding_generator(input_data)
#     assert output.shape == (batch_size, embedding_generator.post_embed_dim)


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
    output = transformer(priors, processed_feat)
    assert output.shape == (bs, group_dim)


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

    output = transformer(input_data)
    assert output.shape == (bs, output_dim)


def test_random_obfuscator():
    bs = 32
    input_dim = 16
    pretraining_ratio = 0.2
    group_matrix = torch.randint(0, 2, size=(5, input_dim)).float()
    obfuscator = RandomObfuscator(pretraining_ratio, group_matrix)
    x = torch.rand((bs, input_dim))

    masked_x, obfuscated_groups, obfuscated_vars = obfuscator(x)
    assert masked_x.shape == x.shape
    assert obfuscated_groups.shape == (bs, group_matrix.shape[0])
    assert obfuscated_vars.shape == x.shape