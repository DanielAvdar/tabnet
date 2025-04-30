import torch

from pytorch_tabnet.tab_network.embedding_generator import EmbeddingGenerator


def test_initialization_with_embeddings(embedding_generator, input_data):
    assert embedding_generator.skip_embedding is False
    assert embedding_generator.post_embed_dim == input_data["input_dim"] + sum(input_data["cat_emb_dims"]) - len(input_data["cat_emb_dims"])
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
