import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from faiss import IndexFlatIP

class Index:

    def __init__(self, d):
        self.index = IndexFlatIP(d)
        self.doc_ids: list[int] = []

    def add(self, ids:list[int], vectors:np.ndarray) -> None:
        if len(ids)!= vectors.shape[0]:
            raise ValueError("number of ids does not match number of vectors")
        elif vectors.ndim != 2:
            raise ValueError("vectors must be 2D nd.array of shape (n, d)")
        self.index.add(vectors)
        self.doc_ids.extend(ids)
    
    def search(self, query: np.ndarray, k: int):
        if query.ndim == 1:
            query = query.reshape(1, -1)

        D, I = self.index.search(query, k)

        results = [[]]

        for i in range(len(D)):
            results.append([])
            for score, id in zip(D[i], I[i]):
                doc_id = self.doc_ids[id]
                results[i].append((float(score), doc_id))
        
        return results


    



