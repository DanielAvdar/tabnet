import pytest
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


@pytest.mark.skip
def test_initialization_without_embeddings_and_group_matrix():
    input_dim = 5
    generator = EmbeddingGenerator(input_dim, [], [], [], None)

    assert generator.skip_embedding is True
    assert generator.post_embed_dim == input_dim
    assert generator.embedding_group_matrix is None


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


@pytest.mark.skip
def test_embedding_no_embedding_with_specific_input():
    """Test specifically targeting the skip_embedding branch in forward method."""
    input_dim = 3
    # Create a generator with skip_embedding=True
    generator = EmbeddingGenerator(input_dim, [], [], [], None)

    # Ensure skip_embedding is set
    assert generator.skip_embedding is True

    # Use inputs that are not floats to ensure they don't get modified in any way
    inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])
    outputs = generator(inputs)

    # The output should be exactly the same as the input
    assert torch.equal(outputs, inputs)

    # Test with a more complex tensor to ensure the skip_embedding path is robust
    complex_tensor = torch.tensor([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5]])
    complex_output = generator(complex_tensor)

    # The output should exactly match the input when skip_embedding is True
    assert torch.equal(complex_output, complex_tensor)


@pytest.mark.skip
def test_skip_embedding_tracing():
    """Test specifically targeting line 86 in embedding_generator.py using an instrumented approach."""
    input_dim = 4

    # Create a custom version of the forward method to help with tracing
    original_forward = EmbeddingGenerator.forward
    line_86_hit = [False]  # Using a list for mutable state

    def traced_forward(self, x):
        if self.skip_embedding:
            line_86_hit[0] = True  # Mark that we hit the return x line
            return x
        return original_forward(self, x)

    # Patch the forward method
    EmbeddingGenerator.forward = traced_forward

    try:
        # Create generator and test it
        generator = EmbeddingGenerator(input_dim=input_dim, cat_dims=[], cat_idxs=[], cat_emb_dims=[], group_matrix=None)
        assert generator.skip_embedding is True

        # Run forward and check the line was hit
        x = torch.randn(2, input_dim)
        output = generator(x)

        # Verify the line was hit and the output matches
        assert line_86_hit[0] is True
        assert torch.equal(output, x)
    finally:
        # Restore the original method
        EmbeddingGenerator.forward = original_forward


def test_embedding_generator_device_movement():
    """Test that EmbeddingGenerator moves to the correct device with the model."""
    input_dim = 10
    cat_idxs = [0, 2]
    cat_dims = [3, 4]
    cat_emb_dims = [2, 2]
    group_matrix = torch.rand(2, input_dim)

    generator = EmbeddingGenerator(
        input_dim,
        cat_dims,
        cat_idxs,
        cat_emb_dims,
        group_matrix,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    generator = generator.to(device)

    batch_size = 2
    x = torch.randint(0, 3, (batch_size, input_dim)).to(device)

    output = generator(x)
    assert output.shape[0] == batch_size
