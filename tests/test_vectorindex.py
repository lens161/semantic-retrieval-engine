import pytest
import os

import numpy as np

from retrieval import embeddings as em
from infrastructure.vectorindex import Index
from unittest.mock import Mock
from faiss import read_index, write_index

from config import VECTOR_DIM

TEST_PATH_1 = "data/testdata/test-index/test1.index"
TEST_PATH_2 = "data/testdata/test-index/test2.index"

@pytest.fixture
def index():
    return Index(VECTOR_DIM, TEST_PATH_1, new=True)

def test_add_vectors(index: Index):
    ids = np.array([10, 11])
    vectors = np.ones((2, VECTOR_DIM), dtype="float32")

    index.add(vectors, ids)

    assert index.size() == 2

def test_add_invalid_shape(index: Index):
    ids = np.array([1])
    vectors = np.ones((VECTOR_DIM,), dtype="float32")

    with pytest.raises(ValueError):
        index.add(vectors, ids)

def test_get_vector(index: Index):
    ids = np.array([1])
    vectors = np.ones((1, VECTOR_DIM), dtype="float32")

    index.add(vectors, ids)

    v = index.get(1)

    assert isinstance(v, np.ndarray)
    assert v.shape[0] == VECTOR_DIM

def test_search_returns_chunk_ids(index: Index):
    ids = np.array([42, 99])
    vectors = np.vstack([
        np.ones(VECTOR_DIM),
        np.zeros(VECTOR_DIM)
    ]).astype("float32")

    index.add(vectors, ids)
    print(vectors)

    query = np.ones(VECTOR_DIM, dtype="float32")

    results = index.search(query, k=1)
    print(results)

    assert isinstance(results, list)
    assert isinstance(results[0][0][1], np.float32)
    assert results[0][0][0] in ids