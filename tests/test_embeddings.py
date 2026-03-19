import pytest
import numpy as np

from processing import embeddings as em
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

def test_chunking(test_files):
    docx, pdf, md, txt = test_files[:4]

    docx_chunks = em.extract_docx(docx)
    pdf_chunks = em.extract_pdf(pdf)
    md_chunks = em.extract_txt_md(md)
    txt_chunks = em.extract_txt_md(txt)
    print(len(docx_chunks))
    print(len(pdf_chunks))
    print(len(md_chunks))
    print(len(txt_chunks))

    # ranges only work for current test files 
    # and for CHUNKSIZE = 30 referring to amount of lines per chunk
    assert len(docx_chunks) > 60 and len(docx_chunks)   < 70
    assert len(pdf_chunks)  > 60 and len(pdf_chunks)    < 70
    assert len(md_chunks)   > 60 and len(md_chunks)     < 70
    assert len(txt_chunks)  > 60 and len(pdf_chunks)    < 70

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

def test_embed_real_model(test_files):
    tf = test_files[:4]

    embeds = [em.embed(f)[1] for f in tf]

    for embeddings in embeds:
        for e in embeddings:
            assert e.shape[0] == 512