# RAG PDF Chat — Streamlit

Same RAG logic as the Node.js/React version, ported module by module to Python.

## Run
```bash
python -m venv venv && venv\Scripts\activate      # Windows (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env                            # then set GROQ_API_KEY
streamlit run app.py
```
Opens at http://localhost:8501. First upload downloads the MiniLM model (~90 MB) once.

## Mapping from the Node project
| Node.js | Python |
|---|---|
| config/index.js | config.py |
| pdfService.js (pdf-parse) | services/pdf_service.py (pypdf) |
| textSplitter.js | services/text_splitter.py (verified: identical chunks) |
| embeddingService.js (Transformers.js) | services/embedding_service.py (sentence-transformers, same model) |
| vectorStore.js / documentRegistry.js | services/vector_store.py / document_registry.py (same JSON files) |
| llmService.js (groq-sdk) | services/llm_service.py (same prompt, temperature 0.1, 1024 tokens) |
| ragService.js | services/rag_service.py |
| Express routes + multer + React | app.py (Streamlit UI, same limits: PDF only, 20 files, 25 MB) |
