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


# def test_query_batch(database: DataBase, conn: sqlite3.Connection):
#     queries = ["airplane", "dog", "car"]
#     database.add_volume(conn)

#     files = search(database, queries, conn)
#     for f in files:
#       print(f)
#       print("\n")

#     assert files != None
#     assert files[0][0].__contains__("texts")
#     assert files[0][0].__contains__("images")
#     assert files[0][0].__contains__("airplane")
#     assert files[1][0].__contains__("animal") 
#     assert files[2][0].__contains__("car")

# def test_single(database: DataBase, conn: sqlite3.Connection):
#     query = "airplanes"
#     print(type(query))
#     database.add_volume(conn)

#     chunks = search(database, query, conn)

#     assert chunks != None 
#     assert chunks[0][1].__contains__("airplane")


    