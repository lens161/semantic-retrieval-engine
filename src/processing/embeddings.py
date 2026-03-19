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
          model: SentenceTransformer = MODEL) -> tuple[str, np.ndarray]:
    filetype = str(from_file(path))

    chunks = []
    if filetype.__contains__("Word"):
        chunks = extract_docx(path)
    elif filetype.__contains__("PDF"):
        chunks = extract_pdf(path)
    elif filetype.__contains__("Unicode text") or filetype.__contains__("ASCII"):
        chunks = extract_txt_md(path)
    elif filetype.__contains__("JPEG") or filetype.__contains__("PNG"):
        img = Image.open(path).convert("RGB")
        chunks = [img]
    print(len(chunks))
    
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    return filetype, embeddings

def extract_docx(path: str) -> list[str]:
    doc = docx.Document(path)
    lines = []
    for p in doc.paragraphs:
        lines.extend(p.text.split("\n"))
    chunks = chunk(lines)
    return chunks

def extract_pdf(path: str) -> list[str]:
    with open(path, 'rb') as f:
        reader = PdfReader(f)
        pages = [p.extract_text() for p in reader.pages]
        lines = []
        for p in pages:
            lines.extend(p.split("\n"))
    return pages
        
def extract_txt_md(path: str) -> list[str]:
    with open(path, 'r') as f:
        lines = f.readlines()
        chunks = chunk(lines)
    return chunks

def chunk(lines):
    size = len(lines)
    chunks = []
    i = 0 
    while i < size:
        if i + CHUNK_SIZE > size:
           chunks.append(str(lines[i:]))
        else:
            chunks.append(str(lines[i:i+CHUNK_SIZE]))
        i += CHUNK_SIZE
    return chunks

def create_query_embeddings(queries: list[str], 
                            model:SentenceTransformer = MODEL) -> np.ndarray:
    vectors = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
    vector_matrix = np.vstack(vectors).astype("float32")
    return vector_matrix