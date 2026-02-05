import numpy as np
import embeddings as em
from index import Index
from sentence_transformers import SentenceTransformer

def main():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2") # pretrained model
    docs = em.load_documents()
    doc_ids, vectors = em.create_doc_embeddings(docs, embedding_model)

    idx = Index(384)

    idx.add(doc_ids, vectors)

    queries = ["where is a dog?", "what do i need for coding?", "what animal meows?"]

    qx = em.create_query_embeddings(queries, embedding_model)
    print(f"queries {qx}")
    print(np.shape(qx))

    results = idx.search(qx, 3)

    print(f"results {results}")

    print(np.array([np.zeros(384), np.ones(384)]))
    print(f"shape = {np.shape(np.array([np.zeros(384), np.ones(384)]))}")
if __name__ == "__main__":
    main()