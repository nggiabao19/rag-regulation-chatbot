import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REGISTRY_FILE = os.path.join(BASE_DIR, "processed_files.json")

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_ID = "BAAI/bge-reranker-base"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "advanced-rag-project"