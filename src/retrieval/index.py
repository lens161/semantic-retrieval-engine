import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from faiss import IndexFlatIP, write_index, read_index
from pathlib import Path

class Index:

    def __init__(self, d:int, index_path:str, new:bool = True):
        self.path = index_path
        idx_path = Path(index_path)
        if not new and idx_path.exists():
            self.index = read_index(index_path)
        else:
            self.index = IndexFlatIP(d)
            idx_path.parent.mkdir(parents=True, exist_ok=True)
            write_index(self.index, index_path)
        self.doc_ids: list[int] = []

    def add(self, ids:list[int], vectors:np.ndarray) -> None:
        if len(ids)!= vectors.shape[0]:
            raise ValueError("number of ids does not match number of vectors")
        elif vectors.ndim != 2:
            raise ValueError("vectors must be 2D nd.array of shape (n, d)")
        elif vectors.dtype != np.dtype("float32"):
            raise ValueError("vectors must be of dtype float32")
        self.index.add(vectors)
        self.doc_ids.extend(ids)

        write_index(self.index, self.path)

    def get(self, i:int) -> np.ndarray:
        return self.index.reconstruct(i)
    
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


    



