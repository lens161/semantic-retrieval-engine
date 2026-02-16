"""
DataBase class.

holds links to indexed volume root and connected SQLite database
Contains helper functions for inserting and retrieving
file metadata from SQLite database.
"""

import os
import sqlite3
import numpy as np

import retrieval.embeddings as em
from sentence_transformers import SentenceTransformer
from .vectorindex import Index

class DataBase:
    def __init__(self, volume_root: str, db: str, 
                 vectorindex: Index):
        self.root = volume_root
        self.db = db
        self.vectorindex = vectorindex
        self.initialise_database()

    def initialise_database(self) -> None:
        conn = sqlite3.connect(self.db)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS file (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            file_type VARCHAR(8) NOT NULL,
            path TEXT UNIQUE NOT NULL
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS chunk (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER
        )
        """)
        conn.commit()
        conn.close()

    def add_volume(self) -> None:
        conn = sqlite3.connect(self.db)
        for dirpath, subdirs, files in os.walk(self.root):
            filepaths = []
            for file in files:
                fp = (file, os.path.join(dirpath, file))
                filepaths.append(fp)
            
            self.add_multiple(filepaths, conn)
        conn.commit()
        conn.close()
            
    def add_multiple(self, files: list[tuple[int, str, str, str]], conn: sqlite3.Connection):
        for f in files:
            path = f[3]
            file_id = f[0]
            chunk_embeds = em.embed(path, file_id)
            try:
                self.add(f, conn, chunk_embeds)
            except sqlite3.IntegrityError:
                continue

    def add(self, file: tuple[int, str, str, str], chunk_embeds:np.ndarray, conn: sqlite3.Connection) -> None:
        file_id = file[0]
        filename = file[1]
        file_type = file[2]
        path = file[3]
        # with conn:
        conn.execute("""INSERT INTO file VALUES(?, ?, ?, ?)""", 
                     (file_id, filename, file_type, path)
                     )
        rows = []
        for i in range(len(chunk_embeds)):
            row = conn.execute("""INSERT INTO chunk (file_id) VALUES(?) RETURNING id""", 
                         (file_id,)).fetchall()
            rows.append(row)
        print(rows)
        chunk_ids = [row[0] for row in rows]
        chunk_ids = [i[0] for i in chunk_ids]

        self.transfer_to_vectorindex(chunk_embeds, chunk_ids)

    def transfer_to_vectorindex(self, chunk_embeds:np.ndarray, chunk_ids:list):
        chunk_ids = np.array(chunk_ids)
        print(chunk_ids)
        self.vectorindex.add(chunk_embeds, chunk_ids)

    def get_file(self, chunk_id: int) -> tuple:
        conn = sqlite3.connect(self.db)
        c = conn.cursor()

        c.execute("""SELECT file_id FROM chunk WHERE id = ?""", (chunk_id, ))
        file = c.fetchall()

        conn.commit()
        conn.close()
        return file
    
    def get_all(self, file_ids: list[int]):
        file_ids = set(file_ids)

        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        c.execute("""SELECT path FROM file WHERE id IN ?""", (file_ids, ))

        file_paths = c.fetchall()

        return file_paths
    
    def get_database(self):
        return self.db