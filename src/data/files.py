"""
File database access layer.

Contains helper functions for inserting and retrieving
file metadata from SQLite database.
"""

import os
import sqlite3

ROOT = "data/testdata/semantic-test-dataset"
DB = 'data/db/test.db'

def crawl(folder: str, db: str) -> None:
    conn = sqlite3.connect(db)
    c = conn.cursor()

    c.execute("""
              CREATE TABLE IF NOT EXISTS file (
                    file_name TEXT NOT NULL,
                    path TEXT UNIQUE NOT NULL
                    ) 
              """)

    for dirpath, subdirs, files in os.walk(folder):
        filepaths = []
        for file in files:
            fp = (file, os.path.join(dirpath, file))
            filepaths.append(fp)
    
        for fp in filepaths:
            try:
                add(fp, c)
            except sqlite3.IntegrityError:
                continue

    conn.commit()
    conn.close()

def add(file: tuple[str, str], c = sqlite3.Cursor) -> None:
    filename = file[0]
    path = file[1]
    c.execute("""INSERT INTO file VALUES(?, ?)""", (filename, path))

def get(file_name: str, conn: sqlite3.Connection) -> tuple:
    c = conn.cursor()
    c.execute("""SELECT * FROM file WHERE file_name = ?""", (file_name, ))
    file = c.fetchall()
    return file


if __name__ == "__main__":
    crawl(ROOT, DB)

    conn = sqlite3.connect(DB)

    c = conn.cursor()

    file = get("airplanes_1.txt", conn)

    print(file)

    conn.execute("""DROP TABLE file""")
    conn.commit()
    conn.close()