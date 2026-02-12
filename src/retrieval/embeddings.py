"""
module for creating embeddings from Data.

Contains helper functions for embedding data and queries.
"""

import json
import numpy as np
import docx
from pypdf import PdfReader

from sentence_transformers import SentenceTransformer
from pathlib import Path
from magic import from_file

from config import TEST_DATA_PATH

DATA_PATH = Path(TEST_DATA_PATH)
CHUNK_SIZE = 10

# def load_documents():
#     with DATA_PATH.open("r", encoding="utf-8") as f:
#         return json.load(f)
    
# def create_doc_embeddings(documents: list[dict], model:SentenceTransformer) -> tuple[list[str], np.ndarray]:
#     vectors = []
#     doc_ids = []

#     for doc in documents:
#         content = doc["metadata"]["synthetic_phrase"]
#         id = doc["id"]
#         v = model.encode(content, convert_to_numpy=True, normalize_embeddings=True) # shape==(384, )
#         vectors.append(v)
#         doc_ids.append(id)
    
#     vector_matrix = np.vstack(vectors).astype("float32") # shape == (len(documents), 384)
#     return doc_ids, vector_matrix

def embed(path: str, doc_id: int, model: SentenceTransformer):
    filetype = str(from_file(path))
    print(filetype)

    chunks = []
    if filetype.__contains__("Word"):
        chunks = extract_docx(path)
    elif filetype.__contains__("PDF"):
        chunks = extract_pdf(path)
    elif filetype.__contains__("ASCII"):
        chunks = extract_txt_md(path)
    
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    return doc_id, embeddings

def extract_docx(path: str) -> list[str]:
    doc = docx.Document(path)
    text = [p for p in doc.paragraphs]
    return text

def extract_pdf(path: str) -> list[str]:
    with open(path, 'r') as f:
        reader = PdfReader(f)
        pages = [p.extract_text() for p in reader.pages]
    return pages
        
def extract_txt_md(path: str) -> list[str]:
    with open(path, 'r') as f:
        lines = f.readlines()
        size = len(lines)
        paragraphs = []
        i = 0 
        while i < size:
            if i + CHUNK_SIZE > size:
               paragraphs.append(str(lines[i:]))
            else:
                paragraphs.append(str(lines[i:i+CHUNK_SIZE]))
            i += CHUNK_SIZE
    return paragraphs

def create_query_embeddings(queries: list[str], model:SentenceTransformer) -> np.ndarray:
    vectors = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
    vector_matrix = np.vstack(vectors).astype("float32")
    return vector_matrix

def main() -> None:
    doc = "data/testdata/semantic_test_dataset/airplanes/texts/airplanes_0.docx"
    pdf = "data/testdata/semantic_test_dataset/airplanes/texts/airplanes_0.pdf"
    md = "data/testdata/semantic_test_dataset/airplanes/texts/airplanes_0.md"
    txt = "data/testdata/semantic_test_dataset/airplanes/texts/airplanes_0.txt"

    embed(doc)
    embed(pdf)
    embed(md)
    embed(txt)

if __name__ =="__main__":

    main()

