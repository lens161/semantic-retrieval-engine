import os
import pytest
import sqlite3
import numpy as np
from unittest.mock import Mock, patch

from infrastructure.database import DataBase
from infrastructure.vectorindex import Index

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

def test_get_file(conn: sqlite3.Connection, database: DataBase):
      database.add_batch(TEST_VALUES_FILE_INPUT, TEST_CHUNK_EMBEDS, conn)

      file1 = database.get_file(2, conn)
      file2 = database.get_file(4, conn)
      file3 = database.get_file(6, conn)
      print(file1)

      assert file1 == TEST_VALUES_FILE_IN_DB[0][0]
      assert file2 == TEST_VALUES_FILE_IN_DB[1][0]
      assert file3 == TEST_VALUES_FILE_IN_DB[2][0]

def test_get_all(conn: sqlite3.Connection, database: DataBase):
      database.add_batch(TEST_VALUES_FILE_INPUT, TEST_CHUNK_EMBEDS, conn)
      file_ids = [2, 3]
      single_id = [1]
      correct_paths = [TEST_VALUES_FILE_IN_DB[1][3], TEST_VALUES_FILE_IN_DB[2][3]]
      correct_single_path = [TEST_VALUES_FILE_IN_DB[0][3]]

      filepaths = database.get_all(file_ids, conn)
      single_path = database.get_all(single_id, conn)
      filepaths = [path[0] for path in filepaths]
      single_path = [single_path[0][0]]

      assert filepaths == correct_paths
      assert single_path == correct_single_path

def test_add_volume():
    
      idx = Index(512, TEST_IDX)
      database = DataBase(TEST_VOLUME, TEST_DB, idx)

      conn = sqlite3.connect(database.get_database())

      database.add_volume(conn)
      files = conn.execute("SELECT * FROM file").fetchall()
      chunks = conn.execute("SELECT * FROM chunk").fetchall()

      conn.execute("""DROP TABLE IF EXISTS file""")
      conn.execute("""DROP TABLE IF EXISTS chunk""")
      conn.close()

      assert len(files) > 0
      assert len(chunks) > 0