import pytest
import numpy as np

from retrieval import embeddings as em
from unittest.mock import Mock

@pytest.fixture
def fake_model():
    fake_model = Mock()
    fake_model.encode.side_effect = [np.zeros(384), np.ones(384)]
    return fake_model

 
TEST_DOCS = [
        {"id": "a", "metadata": {"synthetic_phrase": "first"}},
        {"id": "b", "metadata": {"synthetic_phrase": "second"}},
    ]

def test_load_documents():
    DATA_PATH = "data/testdata/documents.json"

    docs = em.load_documents()

    assert isinstance(docs, list)
    assert all(isinstance(d, dict) for d in docs)
    assert docs[0].get("id") == "doc_018"
    assert docs[2].get("id") == "doc_010"
    assert docs[5].get("id") == "doc_006"

def test_create_doc_embedding_preserves_order(fake_model):

    doc_ids, vectors = em.create_doc_embeddings(TEST_DOCS, fake_model)

    assert doc_ids[0] == "a"
    assert doc_ids[1] == "b"

def test_create_doc_embeddings_correct_shape_and_dtype(fake_model):

    doc_ids, vectors = em.create_doc_embeddings(TEST_DOCS, fake_model)

    assert np.all(vectors[0] == 0)
    assert np.all(vectors[1] == 1)
    assert vectors[0].dtype == np.dtype("float32")
    assert vectors[1].dtype == np.dtype("float32")
    assert vectors[0].shape == (384,)
    assert vectors[1].shape == (384,)

def create_query_embeddings_correct_shape_and_dtype(fake_model):
    queries = ["one", "two"]
    vector_matrix = em.create_query_embeddings(queries, fake_model)

    assert np.all(vector_matrix[0] == 0)
    assert np.all(vector_matrix[1] == 1)
    assert vector_matrix.dtype == np.dtype("float32")
    assert vector_matrix.shape == (2, 384)

    