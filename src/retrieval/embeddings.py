import json
import numpy as np

from sentence_transformers import SentenceTransformer
from pathlib import Path

DATA_PATH = Path("data/testdata/documents.json")

def load_documents():
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)
    
def create_doc_embeddings(documents: list[dict], model:SentenceTransformer) -> tuple[list[str], np.ndarray]:
    vectors = []
    doc_ids = []

    for doc in documents:
        content = doc["metadata"]["synthetic_phrase"]
        id = doc["id"]
        v = model.encode(content, convert_to_numpy=True, normalize_embeddings=True) # shape==(384, )
        vectors.append(v)
        doc_ids.append(id)
    
    vector_matrix = np.vstack(vectors).astype("float32") # shape == (len(documents), 384)
    return doc_ids, vector_matrix

def create_query_embeddings(queries: list[str], model:SentenceTransformer) -> np.ndarray:
    vectors = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
    vector_matrix = np.vstack(vectors).astype("float32")
    return vector_matrix

def main() -> None:
    documents = load_documents()
    embeddings = create_doc_embeddings(documents)

if __name__ =="__main__":
    main()

    

