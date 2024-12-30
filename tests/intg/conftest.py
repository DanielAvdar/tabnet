import pytest
from pathlib import Path
import nbformat
import nbclient

@pytest.fixture
def context_path() -> Path:
    return Path(__file__).parent.parent.parent




def nb_run(nb_name, context_path):
    notebook_path = context_path / nb_name
    nb = nbformat.read(notebook_path, as_version=4)
    client = nbclient.NotebookClient(
        nb,
    )
    client.execute()
