from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_MODEL = "clip-ViT-B-32"
MODEL_PATH = f"{str(BASE_DIR)}/embedding_model/{EMBEDDING_MODEL}"
VECTOR_DIM = 512

# number of lines each text chunk consists of (md and txt)
CHUNK_SIZE = 10