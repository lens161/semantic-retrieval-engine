import numpy as np
import pytest
from psycopg2 import sql
from psycopg2.extensions import connection

from infrastructure.database import DataBase
from processing.embeddings import create_query_embeddings

TEST_VOLUME = "tests/data/semantic_test_dataset"
TEST_DB = 'semantic_test_10'
TEST_DB_FULL = 'semantic_test_512'
TEST_DIM = 10

TEST_VALUES_FILE_INPUT = [("file 1", "doc1", "data/file1"),
                          ("file 2", "doc2", "data/file2"),
                          ("file 3", "doc3", "data/file3")]

TEST_VALUES_FILE_IN_DB = [(1, "file 1", "doc1", "data/file1"),
                          (2, "file 2", "doc2", "data/file2"),
                          (3, "file 3", "doc3", "data/file3")]

TEST_CHUNK_EMBEDS = [np.array([[1,1,1,1,1,1,1,1,1,1],
                              [1,2,2,2,2,2,2,2,2,2],
                              [1,3,3,3,3,3,3,3,3,3]], dtype="float32"),

                     np.array([[2,1,1,1,1,1,1,1,1,1],
                              [2,2,2,2,2,2,2,2,2,2]], dtype="float32"),

                     np.array([[3,1,1,1,1,1,1,1,1,1],
                              [3,2,2,2,2,2,2,2,2,2]], dtype="float32")]

@pytest.fixture()
def database():
      return DataBase(TEST_VOLUME, TEST_DB, TEST_DIM, 
                      "localhost", "test", "test", 5433)

@pytest.fixture()
def database_full():
      return DataBase(TEST_VOLUME, TEST_DB_FULL, 512, 
                      "localhost", "test", "test", 5433)

@pytest.fixture
def conn(database: DataBase):

    conn = database.connect()
    yield conn

    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS chunk CASCADE")
    cur.execute("DROP TABLE IF EXISTS file CASCADE")
    conn.commit()

    conn.close()

    database.initialise_database()

def test_add(conn: connection, database: DataBase):
      c = conn.cursor()
      for file, embeds in zip(TEST_VALUES_FILE_INPUT, TEST_CHUNK_EMBEDS):
            database.add(file, embeds, c)
      c.execute("""SELECT * FROM file""")
      files = c.fetchall()
      c.execute("""SELECT id FROM chunk WHERE file_id=1""")
      embed_ids = c.fetchall()
      embed_ids = [id[0] for id in embed_ids]

      assert TEST_VALUES_FILE_IN_DB[0] in files
      assert TEST_VALUES_FILE_IN_DB[1] in files
      assert TEST_VALUES_FILE_IN_DB[2] in files

      assert embed_ids == [1, 2, 3]

def test_add_batch(conn: connection, database: DataBase):
      c = conn.cursor()
      database.add_batch(TEST_VALUES_FILE_INPUT, TEST_CHUNK_EMBEDS, c)

      c.execute("""SELECT * FROM file""")
      files = c.fetchall()
      c.execute("""SELECT id FROM chunk WHERE file_id=1""")
      embed_ids = c.fetchall()
      embed_ids = [id[0] for id in embed_ids]

      assert TEST_VALUES_FILE_IN_DB[0] in files
      assert TEST_VALUES_FILE_IN_DB[1] in files
      assert TEST_VALUES_FILE_IN_DB[2] in files

      assert embed_ids == [1, 2, 3]

def test_find_chunks(database_full: DataBase):
      db = database_full
      conn = db.connect()
      db.add_volume(conn)

      query = create_query_embeddings(["airplanes"])

      chunks = database_full.find_chunks(query[0], conn, 10)
      conn.close()

      assert chunks != None 
      assert chunks[0][0][3].__contains__("airplane")

def test_find_chunks_multiple_queries(database_full: DataBase):
      db = database_full
      conn = db.connect()
      db.add_volume(conn)

      queries = create_query_embeddings(["airplanes", "dog", "car"])

      chunks = database_full.find_chunks(queries, conn, 10)
      conn.close()

      assert chunks != None 
      assert chunks[0][0][3].__contains__("airplane")
      assert chunks[1][0][3].__contains__("animal")
      assert chunks[2][0][3].__contains__("cars")

def test_images_found(database_full: DataBase):
      db = database_full
      conn = db.connect()
      db.add_volume(conn)

      queries = create_query_embeddings(["airplanes"])

      chunks = database_full.find_chunks(queries, conn, 10)[0]

      conn.close()

      paths = []

      for i in range(len(chunks)):
            paths.append(chunks[i][3])

      print(paths)

      assert any("images" in p for p in paths)



