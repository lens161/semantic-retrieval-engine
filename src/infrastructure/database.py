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

    def add_volume(self, embedding_model) -> None:
        conn = sqlite3.connect(self.db)
        for dirpath, subdirs, files in os.walk(self.root):
            file_list= []
            chunk_embeds = []
            for file in files:
                fp = (file, os.path.join(dirpath, file))
                file_type, embeds = em.embed(fp, embedding_model)
                chunk_embeds.append(embeds)
                file_list.append((file, file_type, fp))
            
            self.add_batch(files, chunk_embeds, conn)
        conn.commit()
        conn.close()
            
    def add_batch(self, files: list[tuple[str, str, str]],
                  chunk_embeds: list[np.ndarray], 
                  conn: sqlite3.Connection) -> None:
        for i, f in enumerate(files):
            try:
                self.add(f, chunk_embeds[i], conn)
            except sqlite3.IntegrityError:
                continue

    def add(self, file: tuple[str, str, str], 
            chunk_embeds: np.ndarray, conn: sqlite3.Connection) -> None:
        filename = file[0]
        file_type = file[1]
        path = file[2]
        file_id = conn.execute(
            """INSERT INTO file (file_name, file_type, path) VALUES(?, ?, ?) RETURNING id""", 
            (filename, file_type, path)).fetchone()[0]
        rows = []
        for i in range(len(chunk_embeds)):
            row = conn.execute(
                """INSERT INTO chunk (file_id) VALUES(?) RETURNING id""", 
                (file_id,)
                ).fetchall()
            
            rows.append(row)

        chunk_ids = [row[0] for row in rows]
        chunk_ids = [i[0] for i in chunk_ids]

        self.transfer_to_vectorindex(chunk_embeds, chunk_ids)

        return file_id

    def transfer_to_vectorindex(self, chunk_embeds:np.ndarray, 
                                chunk_ids:list) -> None:
        chunk_ids = np.array(chunk_ids)
        self.vectorindex.add(chunk_embeds, chunk_ids)

    def get_file(self, chunk_id: int, conn: sqlite3.Connection) -> tuple:

        file = conn.execute(
            """SELECT file_id FROM chunk WHERE id = ?""", (chunk_id, )
            ).fetchone()
        
        return file[0]
    
    def get_all(self, file_ids: list[int], conn: sqlite3.Connection):
        file_ids = list(set(file_ids))
        id_string = ",".join("?" * len(file_ids))
        file_paths = conn.execute(
                f"SELECT path FROM file WHERE id IN ({id_string})", 
                file_ids,
            ).fetchall()

        return file_paths
    
    def get_database(self):
        return self.db
    
