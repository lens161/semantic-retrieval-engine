import os
import pytest
import sqlite3

from infrastrucuture.database import DataBase
from infrastrucuture.vectorindex import Index

TEST_VOLUME = "tests/data/semantic_test_dataset"
TEST_DB = 'tests/data/db/test.db'
TEST_IDX = "tests/data/idx/test.idx"
TEST_DIM = 384

@pytest.fixture
def database():
      idx = Index(TEST_DIM, TEST_IDX)
      db = DataBase(TEST_VOLUME, TEST_DB, idx)

      return db

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
