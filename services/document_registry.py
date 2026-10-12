"""Document metadata registry persisted as JSON (port of documentRegistry.js)."""
import json
from config import config

class DocumentRegistry:
    def __init__(self, file=None):
        self.file = file or config.documents_file
        self.documents = []
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        if self.file.exists():
            raw = self.file.read_text(encoding="utf-8")
            self.documents = json.loads(raw) if raw.strip() else []
        else:
            self.documents = []
        self._loaded = True

    def _persist(self):
        config.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.file.write_text(json.dumps(self.documents, indent=2), encoding="utf-8")

    def add(self, doc):
        self._ensure_loaded()
        self.documents.append(doc)
        self._persist()
        return doc

    def update(self, id, patch):
        self._ensure_loaded()
        for i, d in enumerate(self.documents):
            if d["id"] == id:
                self.documents[i] = {**d, **patch}
                self._persist()
                return self.documents[i]
        return None

    def list(self):
        self._ensure_loaded()
        return self.documents

    def remove(self, id):
        self._ensure_loaded()
        before = len(self.documents)
        self.documents = [d for d in self.documents if d["id"] != id]
        self._persist()
        return before - len(self.documents)

document_registry = DocumentRegistry()
