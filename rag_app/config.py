import os
import sys
from pathlib import Path

class Config:
    SEED=42
    ALLOWED_FILE_EXTENSIONS = set(['.pdf', '.md', '.txt'])

    class Model:
        NAME = "deepseek-r1"
        TEMPERATURE = 0.6

    class Preprocessing:
        CHUNK_SIZE = 2048
        CHUNK_OVERLAP = 128
        EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
        RERANKER = "ms-marco-MiniLM-L-12-v2"
        LLM = "mistral"
        CONTEXTUALIZE_CHUNKS = True
        N_SEMANTIC_RESULTS = 5
        N_BM25_RESULTS = 5
        FAISS_INDEX_PATH = Path("index")

    class Chatbot:
        N_CONTEXT_RESULTS = 5
    
    class Path:
        APP_HOME = Path(os.getenv("APP_HOME", Path(__file__).parent.parent))

    class Database:
        COLLECTION_NAME = "documents"
        CLUSTER_URL = "https://dccbf4cf-85a8-48a9-9a9b-36edad0751d1.europe-west3-0.gcp.cloud.qdrant.io:6333"
