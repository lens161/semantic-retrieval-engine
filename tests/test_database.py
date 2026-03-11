import numpy as np
import pytest
from psycopg2 import sql
from psycopg2.extensions import connection

from infrastructure.database import DataBase
from processing.embeddings import create_query_embeddings

TEST_VOLUME = "tests/data/semantic_test_dataset"
TEST_DB = 'semantic_test'
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
      return DataBase(TEST_VOLUME, TEST_DB, 512, 
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
      embed_ids = c.execute("""SELECT id FROM chunk WHERE file_id=1""")
      embed_ids = c.fetchall()
      embed_ids = [id[0] for id in embed_ids]

      assert TEST_VALUES_FILE_IN_DB[0] in files
      assert TEST_VALUES_FILE_IN_DB[1] in files
      assert TEST_VALUES_FILE_IN_DB[2] in files

      assert embed_ids == [1, 2, 3]

def test_get_file(conn: connection, database: DataBase):
      c = conn.cursor()
      database.add_batch(TEST_VALUES_FILE_INPUT, TEST_CHUNK_EMBEDS, c)

      file1 = database.get_file(2, c)
      file2 = database.get_file(4, c)
      file3 = database.get_file(6, c)
      print(file1)

      assert file1 == TEST_VALUES_FILE_IN_DB[0][0]
      assert file2 == TEST_VALUES_FILE_IN_DB[1][0]
      assert file3 == TEST_VALUES_FILE_IN_DB[2][0]

def test_get_all(conn: connection, database: DataBase):
      c = conn.cursor()
      database.add_batch(TEST_VALUES_FILE_INPUT, TEST_CHUNK_EMBEDS, c)
      chunk_ids = [4, 5, 7]
      single_id = [1]
      correct_paths = [TEST_VALUES_FILE_IN_DB[1][3], TEST_VALUES_FILE_IN_DB[2][3]]
      correct_single_path = TEST_VALUES_FILE_IN_DB[0][3]

      filepaths = database.get_all_files(chunk_ids, conn)
      single_path = database.get_all_files(single_id, conn)
      single_path = single_path[0]

      assert filepaths == correct_paths
      assert single_path == correct_single_path

def test_add_volume(database_full: DataBase):
    
      db = database_full

      conn = db.connect()

      db.add_volume(conn)
      cursor = conn.cursor()
      cursor.execute("SELECT * FROM file")
      files = cursor.fetchall()
      cursor.execute("SELECT * FROM chunk")
      chunks = cursor.fetchall()

      search_result = db.get_file(1, cursor)

      cursor.execute("""DROP TABLE IF EXISTS chunk""")
      cursor.execute("""DROP TABLE IF EXISTS file""")
      conn.commit()
      cursor.close()
      conn.close()

      assert len(files) > 0
      assert len(chunks) > 0
      assert search_result != None

def test_find_chunks(database_full: DataBase):
      db = database_full
      conn = db.connect()
      db.add_volume(conn)

      query = create_query_embeddings(["airplanes"])
      print(type(query))

      chunks = database_full.find_chunks(query, conn, 10)
      conn.close()

      assert chunks != None 
      assert chunks[0][1].__contains__("airplane")