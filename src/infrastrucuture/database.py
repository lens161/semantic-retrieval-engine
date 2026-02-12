"""
DataBase class.

holds links to indexed volume root and connected SQLite database
Contains helper functions for inserting and retrieving
file metadata from SQLite database.
"""

import os
import sqlite3

import retrieval.embeddings as em
from .vectorindex import Index

class DataBase:
    def __init__(self, volume_root: str, db: str, vectorindex: Index):
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
            path TEXT UNIQUE NOT NULL
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS chunk (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER,
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
            
    def add_multiple(self, files: list[tuple[str, str]], conn = sqlite3.Connection):
        for f in files:
            try:
                self.add(f, conn)
            except sqlite3.IntegrityError:
                continue

    def add(file: tuple[str, str], conn = sqlite3.Connection) -> None:
        filename = file[0]
        path = file[1]
        conn.execute("""INSERT INTO file VALUES(?, ?)""", (filename, path))

    def get(self, file_name: str) -> tuple:
        conn = sqlite3.connect(self.db)
        c = conn.cursor()

        c.execute("""SELECT * FROM file WHERE file_name = ?""", (file_name, ))
        file = c.fetchall()

        conn.commit()
        conn.close()
        return file