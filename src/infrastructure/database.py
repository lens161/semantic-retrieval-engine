"""
DataBase class.

holds links to indexed volume root and connected SQLite database
Contains helper functions for inserting and retrieving
file metadata from SQLite database.
"""

import os
import numpy as np
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import connection, cursor
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

import processing.embeddings as em

class DataBase:
    
    def __init__(self, volume_root: str, 
                 db: str, 
                 dim: int,
                 host: str, 
                 user: str,
                 password: str,
                 port: int):
        self.root = volume_root
        self.db = db
        self.dim = dim
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self.initialise_database()

    def initialise_database(self) -> None:

        conn = self.connect()

        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS file (
            id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            file_name TEXT,
            file_type TEXT,
            path TEXT UNIQUE
        )
        """)
        conn.commit()
        chunk_query = sql.SQL("""
        CREATE TABLE IF NOT EXISTS chunk (
            id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            file_id INTEGER REFERENCES file(id),
            embedding vector({})
        )
        """).format(sql.SQL(str(self.dim)))
        cur.execute(chunk_query)
        conn.commit()
        conn.close()

    def add_volume(self, conn: connection, 
                   embedding_model = None) -> None:
        cursor = conn.cursor()
        for dirpath, _, files in os.walk(self.root):
            file_list = []
            chunk_embeds = []
            for file in files:
                full_path = os.path.join(dirpath, file)
                if not embedding_model:
                    file_type, embeds = em.embed(full_path)
                else:
                    file_type, embeds = em.embed(full_path, embedding_model)
                chunk_embeds.append(embeds)
                file_list.append((file, file_type, full_path))
            self.add_batch(file_list, chunk_embeds, cursor)
            conn.commit()
        cursor.close()
            
    def add_batch(self, files: list[tuple[str, str, str]],
                  chunk_embeds: list[np.ndarray], 
                  cursor: cursor) -> None:
        for i, f in enumerate(files):
            try:
                self.add(f, chunk_embeds[i], cursor)
            except psycopg2.IntegrityError:
                continue
            finally:
                cursor.connection.commit()
        cursor.connection.commit()

    def add(self, file: tuple[str, str, str], 
            chunk_embeds: np.ndarray, 
            cursor: cursor):

        filename, file_type, path = file

        cursor.execute(
            """INSERT INTO file (file_name, file_type, path)
               VALUES (%s, %s, %s)
               RETURNING id""",
            (filename, file_type, path)
        )

        file_id = cursor.fetchone()[0]

        chunk_data = [(file_id, embed) for embed in chunk_embeds]

        execute_values(
            cursor,
            """INSERT INTO chunk (file_id, embedding) VALUES %s""",
            chunk_data
        )

        return file_id

    def find_chunks(self, query: np.ndarray, 
                   conn: connection, 
                   k: int = 10) -> list[list[tuple[int, int, float, str]]]:
        
        results = []
        with conn.cursor() as cur:
            if query.ndim == 1:
                results.append(self.query(query, k, cur))
            else:
                for q in query:
                    results.append(self.query(q, k, cur))
        return results

    def query(self, query: np.ndarray, 
              k:int, 
              cur:cursor) -> list[tuple[int, int, float, str]]:
        cur.execute("""
                        SELECT
                            c.id,
                            f.id,
                            1 - (c.embedding <=> %s) AS similarity,
                            f.path
                        FROM chunk c
                        JOIN file f ON c.file_id = f.id
                        ORDER BY c.embedding <=> %s
                        LIMIT %s
                        """, 
                        (query, query, k))
        chunks = cur.fetchall()
        return chunks
    
    def get_database(self):
        return self.db

    def connect(self) -> connection:
        conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.db,
            user=self.user,
            password=self.password
        )
        register_vector(conn)
        return conn

    def disconnect(self, conn: connection):
        conn.commit()
        conn.close()