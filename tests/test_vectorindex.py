import pytest
import os

import numpy as np

from retrieval import embeddings as em
from infrastrucuture.vectorindex import Index
from unittest.mock import Mock
from faiss import read_index, write_index

from config import VECTOR_DIM

TEST_PATH_1 = "data/testdata/test-index/test1.index"
TEST_PATH_2 = "data/testdata/test-index/test2.index"

@pytest.fixture
def index():
    return Index(VECTOR_DIM, TEST_PATH_1, new=True)


def test_add_vectors(index):
    ids = [10, 11]
    vectors = np.ones((2, VECTOR_DIM), dtype="float32")

    index.add(ids, vectors)

    assert len(index.doc_ids) == 2


def test_add_invalid_shape(index):
    ids = [1]
    vectors = np.ones((VECTOR_DIM,), dtype="float32")

    with pytest.raises(ValueError):
        index.add(ids, vectors)


def test_get_vector(index):
    ids = [1]
    vectors = np.ones((1, VECTOR_DIM), dtype="float32")

    index.add(ids, vectors)

    v = index.get(0)

    assert isinstance(v, np.ndarray)
    assert v.shape[0] == VECTOR_DIM


def test_search_returns_doc_ids(index):
    ids = [42, 99]
    vectors = np.vstack([
        np.ones(VECTOR_DIM),
        np.zeros(VECTOR_DIM)
    ]).astype("float32")

    index.add(ids, vectors)

    query = np.ones(VECTOR_DIM, dtype="float32")

    results = index.search(query, k=1)

    assert isinstance(results, list)
    assert isinstance(results[0][0][1], int)
    assert results[0][0][1] in ids