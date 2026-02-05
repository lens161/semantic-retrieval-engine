import numpy as np
import embeddings as em
from index import Index
from sentence_transformers import SentenceTransformer

def main():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2") # pretrained model
    docs = em.load_documents()
    doc_ids, vectors = em.create_doc_embeddings(docs)

    idx = Index(384)

    idx.add(doc_ids, vectors)

    queries = ["where is a dog?", "what do i need for coding?", "what animal meows?"]

    qx = em.create_query_embeddings(queries)

    results = idx.search(qx, 3)


if __name__ == "__main__":
    main()