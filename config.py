"""Environment-driven configuration (port of backend/src/config/index.js)."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
# Optional: point DATA_DIR at the Node project's backend/data to reuse its uploads / vector store.
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))

class Config:
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    embedding_model = os.getenv("EMBEDDING_MODEL", "Xenova/all-MiniLM-L6-v2")

    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))

    top_k = int(os.getenv("TOP_K", "5"))
    min_score = float(os.getenv("MIN_SCORE", "0.2"))

    uploads_dir = DATA_DIR / "uploads"
    vector_store_dir = DATA_DIR / "vector_store"
    vector_store_file = DATA_DIR / "vector_store" / "store.json"
    documents_file = DATA_DIR / "vector_store" / "documents.json"

    # multer limits from upload.middleware.js
    max_file_size = 25 * 1024 * 1024
    max_files = 20

config = Config()
