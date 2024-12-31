import torch
import pytest
from pytorch_tabnet.tab_network import (
    AttentiveTransformer,
    FeatTransformer,
    EmbeddingGenerator,
    RandomObfuscator, TabNetPretraining,
)


@pytest.fixture
def embedding_generator():
    input_dim = 10
    cat_dims = [2, 3, 4]
    cat_idxs = [0, 2, 4]
    cat_emb_dims = [1, 2, 3]

    group_matrix = torch.randint(0, 2, size=(3, input_dim)).float()

    return EmbeddingGenerator(input_dim, cat_dims, cat_idxs, cat_emb_dims, group_matrix)


@pytest.fixture
def input_data():
    return {
        "input_dim": 10,
        "cat_dims": [5, 10],
        "cat_idxs": [1, 3],
        "cat_emb_dims": [3, 4],
        "group_matrix": torch.rand(2, 10)
    }


@pytest.fixture
def embedding_generator(input_data):
    return EmbeddingGenerator(
        input_data["input_dim"],
        input_data["cat_dims"],
        input_data["cat_idxs"],
        input_data["cat_emb_dims"],
        input_data["group_matrix"]
    )


def test_initialization_with_embeddings(embedding_generator, input_data):
    assert embedding_generator.skip_embedding is False
    assert embedding_generator.post_embed_dim == input_data["input_dim"] + sum(input_data["cat_emb_dims"]) - len(
        input_data["cat_emb_dims"])
    assert len(embedding_generator.embeddings) == len(input_data["cat_dims"])


def test_initialization_without_embeddings():
    input_dim = 5
    group_matrix = torch.rand(2, 5)
    generator = EmbeddingGenerator(input_dim, [], [], [], group_matrix)

    assert generator.skip_embedding is True
    assert generator.post_embed_dim == input_dim
    assert generator.embedding_group_matrix.shape == group_matrix.shape


def test_forward_with_embeddings(embedding_generator, input_data):
    batch_size = 8
    x = torch.randint(0, 5, (batch_size, input_data["input_dim"]))

    output = embedding_generator(x)

    assert output.shape == (batch_size, embedding_generator.post_embed_dim)


def test_forward_without_embeddings():
    input_dim = 5
    group_matrix = torch.rand(2, 5)
    generator = EmbeddingGenerator(input_dim, [], [], [], group_matrix)

    batch_size = 8
    x = torch.rand(batch_size, input_dim)

    output = generator(x)

    assert torch.allclose(output, x)


def test_embedding_group_matrix_update(embedding_generator, input_data):
    group_matrix = input_data["group_matrix"]
    expected_shape = (group_matrix.shape[0], embedding_generator.post_embed_dim)

    assert embedding_generator.embedding_group_matrix.shape == expected_shape


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


@pytest.mark.parametrize(
    "input_dim, pretraining_ratio, n_steps, n_independent, n_shared, expected_error",
    [
        (10, 0.2, 0, 2, 2, "n_steps should be a positive integer."),
        (10, 0.2, 3, 0, 0, "n_shared and n_independent can't be both zero."),
    ],
)
def test_tabnet_pretraining_initialization_errors(
    input_dim, pretraining_ratio, n_steps, n_independent, n_shared, expected_error
):
    with pytest.raises(ValueError, match=expected_error):
        TabNetPretraining(
            input_dim=input_dim,
            pretraining_ratio=pretraining_ratio,
            n_steps=n_steps,
            n_independent=n_independent,
            n_shared=n_shared,
        )