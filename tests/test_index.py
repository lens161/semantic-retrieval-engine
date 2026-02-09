import pytest
import os

import numpy as np

from retrieval import embeddings as em
from retrieval.index import Index
from unittest.mock import Mock
from faiss import read_index, write_index

from retrieval.config import VECTOR_DIM

TEST_PATH_1 = "data/testdata/test-index/test1.index"
TEST_PATH_2 = "data/testdata/test-index/test2.index"

@pytest.fixture
def fake_model():
    fake_model = Mock()
    fake_model.encode.side_effect = [np.zeros(384), np.ones(384)]
    return fake_model

@pytest.fixture
def index():
    return Index(VECTOR_DIM, TEST_PATH_1)

TEST_DOCS = [
        {"id": "a", "metadata": {"synthetic_phrase": "first"}},
        {"id": "b", "metadata": {"synthetic_phrase": "second"}},
    ]

def test_correct_vectors_added(fake_model, index):
    doc_ids, vectors = em.create_doc_embeddings(TEST_DOCS, fake_model)

    index.add(doc_ids, vectors)

    assert np.all(index.get(0) == 0) 
    assert np.all(index.get(1) == 1)

def test_add_wrong_input_throws(index, fake_model):
    doc_ids, vectors = em.create_doc_embeddings(TEST_DOCS, fake_model)

    wrong_ids = ["one", "two", "three"]
    wrong_vectors = np.zeros((10, 10, 10), dtype="float32")
    wrong_vectors_dtype = vectors.astype("float16")
    with pytest.raises(ValueError):
        index.add(wrong_ids, vectors)
    with pytest.raises(ValueError):
        index.add(doc_ids, wrong_vectors)
    with pytest.raises(ValueError):
        index.add(doc_ids, wrong_vectors_dtype)

def test_search_returns_correct_results(index, fake_model):

    doc_ids, vectors = em.create_doc_embeddings(TEST_DOCS, fake_model)
    query_matrix = np.array([np.zeros(384), np.ones(384)])

    index.add(doc_ids, vectors)
    results = index.search(query_matrix, 1)

    assert results[0][0][1] == "a"
    assert results[1][0][1] == "b"

def test_index_writes_correctly():
    idx = Index(VECTOR_DIM, TEST_PATH_2)

    assert os.path.exists(TEST_PATH_2)

def test_index_loaded_correctly(fake_model):
    doc_ids, vectors = em.create_doc_embeddings(TEST_DOCS, fake_model)
    idx = Index(VECTOR_DIM, TEST_PATH_1, False)

    assert np.allclose(idx.get(0), vectors[0])
    assert np.allclose(idx.get(1), vectors[1])


    



