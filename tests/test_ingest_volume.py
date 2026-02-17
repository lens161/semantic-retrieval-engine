import numpy as np
from unittest.mock import Mock, patch

from infrastructure.database import DataBase
from infrastructure.vectorindex import Index
from config import VOLUME_ROOT, VECTOR_DIM, DB, INDEX_PATH

TEST_DB = 'tests/data/db/test.db'


def test_ingest_volume():
    idx = Index(VECTOR_DIM, INDEX_PATH)
    db = DataBase(VOLUME_ROOT, TEST_DB, idx)

    db.add_volume()

    conn = db.connect()

    count = conn.execute("""SELECT COUNT(*) from file""").fetchall()

    assert count[0][0] > 10 