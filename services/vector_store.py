"""JSON-backed vector store with cosine similarity (port of vectorStore.js).
Record: { id, text, embedding: [...], metadata: {...} }"""
import json, math
from config import config

class VectorStore:
    def __init__(self, store_file=None):
        self.store_file = store_file or config.vector_store_file
        self.records = []
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        if self.store_file.exists():
            raw = self.store_file.read_text(encoding="utf-8")
            self.records = json.loads(raw) if raw.strip() else []
        else:
            self.records = []
        self._loaded = True

    def _persist(self):
        config.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.store_file.write_text(json.dumps(self.records), encoding="utf-8")

    def add_documents(self, items):
        self._ensure_loaded()
        self.records.extend(items)
        self._persist()
        return len(items)

    @staticmethod
    def cosine_similarity(a, b):
        dot = norm_a = norm_b = 0.0
        for i in range(len(a)):
            dot += a[i] * b[i]
            norm_a += a[i] * a[i]
            norm_b += b[i] * b[i]
        if norm_a == 0 or norm_b == 0:
            return 0
        return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))

    def similarity_search(self, query_embedding, top_k=5, min_score=0, document_id=None):
        self._ensure_loaded()
        pool = [r for r in self.records if r["metadata"].get("documentId") == document_id] if document_id else self.records
        scored = [{**r, "score": VectorStore.cosine_similarity(query_embedding, r["embedding"])} for r in pool]
        kept = sorted((r for r in scored if r["score"] >= min_score), key=lambda r: -r["score"])[:top_k]
        return [{k: v for k, v in r.items() if k != "embedding"} for r in kept]  # don't ship raw vector

    def remove_by_document_id(self, document_id):
        self._ensure_loaded()
        before = len(self.records)
        self.records = [r for r in self.records if r["metadata"].get("documentId") != document_id]
        self._persist()
        return before - len(self.records)

    def count(self):
        self._ensure_loaded()
        return len(self.records)

    def clear(self):
        self.records = []
        self._persist()

vector_store = VectorStore()
