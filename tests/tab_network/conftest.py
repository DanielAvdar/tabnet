import pytest
import torch


@pytest.fixture
def input_data():
    return {
        "input_dim": 10,
        "cat_dims": [5, 10],
        "cat_idxs": [1, 3],
        "cat_emb_dims": [3, 4],
        "group_matrix": torch.rand(2, 10),
    }


@pytest.fixture
def embedding_generator(input_data):
    from pytorch_tabnet.tab_network.embedding_generator import EmbeddingGenerator

    return EmbeddingGenerator(
        input_data["input_dim"],
        input_data["cat_dims"],
        input_data["cat_idxs"],
        input_data["cat_emb_dims"],
        input_data["group_matrix"],
    )
