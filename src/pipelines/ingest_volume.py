"""
Ingest Pipeline for volumes.

recursively crawls the volume for (unknown) files and adds them
to the vector index / database infrastructure
"""
import retrieval.embeddings as em

from config import VOLUME_ROOT, DB, EMBEDDING_MODEL, VECTOR_DIM, INDEX_PATH
from infrastrucuture.vectorindex import Index
from infrastrucuture.database import DataBase

idx = Index(VECTOR_DIM, INDEX_PATH)
db = DataBase(VOLUME_ROOT, DB, idx)




