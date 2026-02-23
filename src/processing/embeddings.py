"""
Embeddings module for creating embeddings from files.

Contains helper functions for reading, chunking and embedding data and queries.
"""
import numpy as np
import os
import docx
from pathlib import Path
from pypdf import PdfReader
from PIL import Image

from sentence_transformers import SentenceTransformer
from magic import from_file

from config import EMBEDDING_MODEL, MODEL_PATH, CHUNK_SIZE

if not os.path.exists(MODEL_PATH):
    path = Path(MODEL_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    MODEL = SentenceTransformer(EMBEDDING_MODEL)
    MODEL.save(MODEL_PATH)
else:
    MODEL = SentenceTransformer(MODEL_PATH)

def embed(path: str, 
          model: SentenceTransformer = MODEL) -> tuple[int, np.ndarray]:
    filetype = str(from_file(path))

    chunks = []
    if filetype.__contains__("Word"):
        chunks = extract_docx(path)
    elif filetype.__contains__("PDF"):
        chunks = extract_pdf(path)
    elif filetype.__contains__("ASCII"):
        chunks = extract_txt_md(path)
    elif filetype.__contains__("JPEG") or filetype.__contains__("PNG"):
        chunks = [Image.open(path)]
    
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    return filetype, embeddings

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

def create_query_embeddings(queries: list[str], 
                            model:SentenceTransformer = MODEL) -> np.ndarray:
    vectors = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
    vector_matrix = np.vstack(vectors).astype("float32")
    return vector_matrix