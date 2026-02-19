import pytest
import numpy as np

from retrieval import embeddings as em
from unittest.mock import Mock

from config import VECTOR_DIM

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
        "tests/data/docs/000000005477.jpg",
        "tests/data/docs/000000052412.jpg",
        "tests/data/docs/000000052412.png",
        "tests/data/docs/000000096549.png"
    ]

def test_embed_correct_shape_and_dtype(fake_model, test_files):
    for file in test_files:
        _, embeddings = em.embed(str(file), fake_model)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[1] == VECTOR_DIM
        assert embeddings.dtype == np.float32

def test_embed_with_real_model_shape_and_dtype(test_files):
    for file in test_files:
        _, embeddings = em.embed(str(file))

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[1] == VECTOR_DIM
        assert embeddings.dtype == np.float32

def test_create_query_embeddings_correct_shape_and_dtype(fake_model):
    queries = ["one", "two"]

    vectors = em.create_query_embeddings(queries, fake_model)

    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (2, VECTOR_DIM)
    assert vectors.dtype == np.float32