import pytest
import numpy as np

from retrieval import embeddings as em
from unittest.mock import Mock
from pathlib import Path

from config import VECTOR_DIM

TEST_DATA_DIR = Path("tests/data/docs")

@pytest.fixture
def fake_model():
    fake_model = Mock()

    def fake_encode(inputs, convert_to_numpy=True, normalize_embeddings=True):
        n = len(inputs)
        return np.ones((n, VECTOR_DIM), dtype="float32")

    fake_model.encode.side_effect = fake_encode
    return fake_model


@pytest.fixture
def test_files():
    return [
        "tests/data/docs/0.docx",
        "tests/data/docs/0.pdf",
        "tests/data/docs/0.md",
        "tests/data/docs/0.txt",
    ]


@pytest.mark.parametrize("doc_id", [0, 3, 161])
def test_embed_correct_shape_and_dtype(fake_model, test_files, doc_id):
    for file in test_files:
        returned_id, embeddings = em.embed(str(file), doc_id, fake_model)

        assert returned_id == doc_id
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[1] == VECTOR_DIM
        assert embeddings.dtype == np.float32


def test_create_query_embeddings_correct_shape_and_dtype(fake_model):
    queries = ["one", "two"]

    vectors = em.create_query_embeddings(queries, fake_model)

    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (2, VECTOR_DIM)
    assert vectors.dtype == np.float32