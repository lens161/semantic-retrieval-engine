import os
import pytest
import sqlite3

from infrastructure.database import DataBase
from infrastructure.vectorindex import Index
from retrieval import search

from config import VECTOR_DIM

TEST_VOLUME = "tests/data/semantic_test_dataset"
TEST_DB = 'tests/data/db/test.db'
TEST_IDX = "tests/data/idx/test.idx"

@pytest.fixture
def database():
      idx = Index(VECTOR_DIM, TEST_IDX)
      db = DataBase(TEST_VOLUME, TEST_DB, idx)
      yield db

      os.remove(TEST_IDX)

@pytest.fixture
def conn(database: DataBase):
      conn = sqlite3.connect(database.get_database())
      yield conn # tests are run with this state of the database

      # runs after tests to clean up:
      conn.execute("""DROP TABLE IF EXISTS file""")
      conn.execute("""DROP TABLE IF EXISTS chunk""")

      conn.close()

def test_search_batch(database: DataBase, conn: sqlite3.Connection):
    queries = ["airplane", "canine", "street"]
    database.add_volume(conn)

    files = search.search(database, queries, conn)

    assert files != None
    assert files[0][0].__contains__("airplane")
    # animal and car images are mixed. so any first result containing an animal or car is fine
    assert files[1][0].__contains__("animal") or files[1][0].__contains__("car")
    assert files[2][0].__contains__("car") or files[2][0].__contains__("animal")

def test_single(database: DataBase, conn: sqlite3.Connection):
    query = "airplanes"
    print(type(query))
    database.add_volume(conn)

    files = search.search(database, query, conn)

    assert files != None 
    assert files[0][0].__contains__("airplane")


    