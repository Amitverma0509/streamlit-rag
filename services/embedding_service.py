"""Singleton local embedding model (port of embeddingService.js).
Same all-MiniLM-L6-v2 model, mean pooling + L2 normalisation, 384-d vectors."""
import threading
from config import config

def _py_model_name(name):
    # Xenova/* is the ONNX port used by Transformers.js; the original weights live under sentence-transformers/*
    return name.replace("Xenova/", "sentence-transformers/", 1) if name.startswith("Xenova/") else name

class EmbeddingManager:
    def __init__(self, model_name=None):
        self.model_name = model_name or config.embedding_model
        self.extractor = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self.extractor is None:
                print(f'[embeddings] loading model "{self.model_name}" (first run downloads the weights)...')
                from sentence_transformers import SentenceTransformer
                self.extractor = SentenceTransformer(_py_model_name(self.model_name))
                print("[embeddings] model ready")
        return self.extractor

    def embed_text(self, text):
        extractor = self._load()
        return extractor.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts):
        extractor = self._load()
        return [extractor.encode(t, normalize_embeddings=True).tolist() for t in texts]  # sequential, as original

embedding_manager = EmbeddingManager()
