import os

class Config:
    EMBEDDING_MODEL_URL = os.getenv('EMBEDDING_MODEL_URL', 'http://embedding_model:8080')
    GENERATION_MODEL_URL = os.getenv('GENERATION_MODEL_URL', 'http://generation_model:8081')
    DATA_DIR = os.getenv('DATA_DIR', 'local_data')
    VECTOR_STORE_PATH = os.getenv('VECTOR_STORE_PATH', 'faiss_index')
    CHUNK_SIZE = os.getenv('CHUNK_SIZE', 10)
    CHUNK_OVERLAP = os.getenv('CHUNK_OVERLAP', 5)