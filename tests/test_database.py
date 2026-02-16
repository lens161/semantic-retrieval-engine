import os
import pytest
import sqlite3
import numpy as np

from infrastrucuture.database import DataBase
from infrastrucuture.vectorindex import Index

TEST_VOLUME = "tests/data/semantic_test_dataset"
TEST_DB = 'tests/data/db/test.db'
TEST_IDX = "tests/data/idx/test.idx"
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

@pytest.fixture
def database():
      idx = Index(TEST_DIM, TEST_IDX)
      db = DataBase(TEST_VOLUME, TEST_DB, idx)
      yield db

      os.remove(TEST_IDX)

@pytest.fixture
def conn(database: DataBase):
      conn = sqlite3.connect(database.get_database())
      yield conn # tests are run with this state of the connected database

      # runs after tests to clean up:
      conn.execute("""DROP TABLE IF EXISTS file""")
      conn.execute("""DROP TABLE IF EXISTS chunk""")

      conn.close()

def test_correct_initialisation_of_tables(conn: sqlite3.Connection):
      c = conn.cursor()
      c.execute("""SELECT name FROM sqlite_master WHERE type='table'
                AND name NOT LIKE 'sqlite_%' """)
      tables = [row[0] for row in c.fetchall()]
      c.execute("""PRAGMA table_info(file)""")
      file_table_info = c.fetchall()
      file_columns = [c[1] for c in file_table_info]
      file_columns_types = [c[2] for c in file_table_info]
      c.execute("""PRAGMA table_info(chunk)""")
      chunk_table_info = c.fetchall()
      chunk_columns = [c[1] for c in chunk_table_info]
      chunk_columns_types = [c[2] for c in chunk_table_info]

      assert "file" in tables
      assert "chunk" in tables

      assert "id" in file_columns
      assert "file_name" in file_columns
      assert "path" in file_columns
      assert file_columns_types[0] == "INTEGER"
      assert file_columns_types[1] == "TEXT"
      assert "VARCHAR" in file_columns_types[2]
      assert file_columns_types[3] == "TEXT"

      assert "id" in chunk_columns
      assert "file_id" in chunk_columns
      assert chunk_columns_types[0] == "INTEGER"
      assert chunk_columns_types[1] == "INTEGER"

def test_transfer_to_vectorindex(database: DataBase):
      database.transfer_to_vectorindex(TEST_CHUNK_EMBEDS[0], [1, 2, 3])
      database.transfer_to_vectorindex(TEST_CHUNK_EMBEDS[1], [4, 5])
      database.transfer_to_vectorindex(TEST_CHUNK_EMBEDS[2], [100, 200])
      assert np.array_equal(TEST_CHUNK_EMBEDS[0][0], database.vectorindex.get(1))
      assert np.array_equal(TEST_CHUNK_EMBEDS[0][1], database.vectorindex.get(2))
      assert np.array_equal(TEST_CHUNK_EMBEDS[0][2], database.vectorindex.get(3))
      assert np.array_equal(TEST_CHUNK_EMBEDS[1][0], database.vectorindex.get(4))
      assert np.array_equal(TEST_CHUNK_EMBEDS[1][1], database.vectorindex.get(5))
      assert np.array_equal(TEST_CHUNK_EMBEDS[2][0], database.vectorindex.get(100))
      assert np.array_equal(TEST_CHUNK_EMBEDS[2][1], database.vectorindex.get(200))

def test_add(conn: sqlite3.Connection, database: DataBase):

      for file, embeds in zip(TEST_VALUES_FILE_INPUT, TEST_CHUNK_EMBEDS):
            database.add(file, embeds, conn)
      files = conn.execute("""SELECT * FROM file""").fetchall()
      embed_ids = conn.execute("""SELECT id FROM chunk WHERE file_id=1""").fetchall()
      embed_ids = [id[0] for id in embed_ids]

      assert TEST_VALUES_FILE_IN_DB[0] in files
      assert TEST_VALUES_FILE_IN_DB[1] in files
      assert TEST_VALUES_FILE_IN_DB[2] in files

      assert embed_ids == [1, 2, 3]

def test_add_batch(conn: sqlite3.Connection, database: DataBase):
      database.add_batch(TEST_VALUES_FILE_INPUT, TEST_CHUNK_EMBEDS, conn)

      files = conn.execute("""SELECT * FROM file""").fetchall()
      embed_ids = conn.execute("""SELECT id FROM chunk WHERE file_id=1""").fetchall()
      embed_ids = [id[0] for id in embed_ids]

      assert TEST_VALUES_FILE_IN_DB[0] in files
      assert TEST_VALUES_FILE_IN_DB[1] in files
      assert TEST_VALUES_FILE_IN_DB[2] in files

      assert embed_ids == [1, 2, 3]

      