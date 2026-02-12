import pytest
import sqlite3

TEST_VOLUME = "data/testdata/semantic-test-dataset"
TEST_DB = 'data/db/test.db'

def test_add():
    conn = sqlite3.connect(TEST_DB)
    c = conn.cursor()
    c.execute("""
              CREATE TABLE IF NOT EXISTS file (
                    file_name TEXT NOT NULL,
                    path TEXT UNIQUE NOT NULL
                    ) 
              """)

