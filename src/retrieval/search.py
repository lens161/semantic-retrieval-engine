"""
Search Engine

receives a query / List of queries and returns files that semantically match the query
"""
from infrastructure.database import DataBase
from processing.embeddings import create_query_embeddings

from psycopg2.extensions import connection

def search(database: DataBase,
           query: str | list[str],
           conn: connection) -> list[str]:

    if isinstance(query, str):
        queries = [query]
    elif isinstance(query, list):
        queries = query
    else:
        raise TypeError("queries must be of type str or list[str]")
    
    
    
