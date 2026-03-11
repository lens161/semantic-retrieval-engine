# import os
# import pytest
# import sqlite3

# from infrastructure.database import DataBase
# from infrastructure.vectorindex import Index
# from retrieval.search import search

# from config import VECTOR_DIM

# TEST_VOLUME = "tests/data/semantic_test_dataset"
# TEST_DB = 'tests/data/db/test.db'
# TEST_IDX = "tests/data/idx/test.idx"


# @pytest.fixture(scope="session")
# def conn():

#     conn = psycopg2.connect(
#         host="localhost",
#         dbname="semantic_test",
#         user="user",
#         password="password"
#     )

#     yield conn

#     conn.close()

# @pytest.fixture(scope="session", autouse=True)
# def init_db(conn):

#     cur = conn.cursor()

#     cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

#     cur.execute("""
#     CREATE TABLE IF NOT EXISTS files (
#         id SERIAL PRIMARY KEY,
#         file_name TEXT,
#         file_type TEXT,
#         path TEXT UNIQUE
#     )
#     """)

#     cur.execute("""
#     CREATE TABLE IF NOT EXISTS chunks (
#         id SERIAL PRIMARY KEY,
#         file_id INTEGER REFERENCES files(id),
#         embedding vector(512)
#     )
#     """)

#     conn.commit()

# def test_query_batch(database: DataBase, conn: sqlite3.Connection):
#     queries = ["airplane", "dog", "car"]
#     database.add_volume(conn)

#     files = search(database, queries, conn)
#     for f in files:
#       print(f)
#       print("\n")

#     assert files != None
#     assert files[0][0].__contains__("texts")
# #     assert files[0][0].__contains__("images")
#     assert files[0][0].__contains__("airplane")
#     assert files[1][0].__contains__("animal") 
#     assert files[2][0].__contains__("car")

# def test_single(database: DataBase, conn: sqlite3.Connection):
#     query = "airplanes"
#     print(type(query))
#     database.add_volume(conn)

#     files = search(database, query, conn)

#     assert files != None 
#     assert files[0][0].__contains__("airplane")


    