"""
Search Engine

receives a query and returns files that semantically match the query
"""
from infrastructure.database import DataBase
from infrastructure.vectorindex import Index
from retrieval.embeddings import create_query_embeddings

from config import VECTOR_DIM, DB, VOLUME_ROOT, INDEX_PATH

from sqlite3 import Connection

def search(database: DataBase, query: str | list[str], conn: Connection) -> list[str]:

    if isinstance(query, str):
        queries = [query]
    elif isinstance(query, list):
        queries = query
    else:
        raise TypeError("queries must be of type str or list[str]")
    
    query_embeddings = create_query_embeddings(queries)

    results = database.vectorindex.search(query_embeddings, 20)

    chunk_ids = [[i for i, _ in r] for r in results]

    files = [database.get_all_files(per_query_ids, conn)
             for per_query_ids in chunk_ids]
    return files

if __name__ == "__main__":

    queries = ["airplane", "canine", "drive"]
    idx = Index(VECTOR_DIM, INDEX_PATH)
    db = DataBase(VOLUME_ROOT, DB, idx)

    conn = db.connect()
    db.add_volume(conn)

    files = search(db, queries, conn)

    print(files)







