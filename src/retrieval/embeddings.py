"""
module for creating embeddings from Data.

Contains helper functions for embedding data and queries.
"""
import numpy as np
import docx
from pypdf import PdfReader

from sentence_transformers import SentenceTransformer
from pathlib import Path
from magic import from_file

from config import TEST_DATA_PATH, EMBEDDING_MODEL

DATA_PATH = Path(TEST_DATA_PATH)
CHUNK_SIZE = 10

MODEL = SentenceTransformer(EMBEDDING_MODEL)

def embed(path: str, file_id: int, model: SentenceTransformer = MODEL) -> tuple[int, np.ndarray]:
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

    return file_id, embeddings

def extract_docx(path: str) -> list[str]:
    doc = docx.Document(path)
    text = [p.text for p in doc.paragraphs]
    return text

def extract_pdf(path: str) -> list[str]:
    with open(path, 'rb') as f:
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

def create_query_embeddings(queries: list[str], model:SentenceTransformer = MODEL) -> np.ndarray:
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

