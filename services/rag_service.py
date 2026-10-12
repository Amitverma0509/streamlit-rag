"""Orchestrates both pipelines (port of ragService.js)."""
import uuid
from datetime import datetime, timezone
from config import config
from services.pdf_service import extract_text_from_pdf
from services.text_splitter import split_text
from services.embedding_service import embedding_manager
from services.vector_store import vector_store
from services.document_registry import document_registry
from services.llm_service import generate_answer

def ingest_pdf(file_path, original_name):
    """DATA INGESTION PIPELINE: PDF -> extract text -> chunk -> embed -> store in vector DB"""
    document_id = str(uuid.uuid4())

    doc = document_registry.add({
        "id": document_id,
        "filename": original_name,
        "status": "processing",
        "numPages": 0,
        "numChunks": 0,
        "uploadedAt": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
    })

    try:
        res = extract_text_from_pdf(file_path)
        text, num_pages = res["text"], res["numPages"]

        chunks = split_text(text, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)

        if not chunks:
            document_registry.update(document_id, {"status": "empty", "numPages": num_pages})
            return {**doc, "status": "empty", "numPages": num_pages, "numChunks": 0}

        embeddings = embedding_manager.embed_batch(chunks)

        records = [{
            "id": f"{document_id}-{i}",
            "text": chunk_text,
            "embedding": embeddings[i],
            "metadata": {"documentId": document_id, "filename": original_name, "chunkIndex": i},
        } for i, chunk_text in enumerate(chunks)]

        vector_store.add_documents(records)

        return document_registry.update(document_id, {"status": "ready", "numPages": num_pages, "numChunks": len(chunks)})
    except Exception as err:
        document_registry.update(document_id, {"status": "failed", "error": str(err)})
        raise

def ingest_multiple_pdfs(files):
    """files: list of (file_path, original_name). Sequential to keep memory/CPU predictable."""
    return [ingest_pdf(path, name) for path, name in files]

def ask_question(question, top_k=None, min_score=None, document_id=None):
    """RETRIEVAL + AUGMENTATION + GENERATION: question -> embed -> search -> context -> LLM -> answer"""
    top_k = config.top_k if top_k is None else top_k
    min_score = config.min_score if min_score is None else min_score
    if not question or not question.strip():
        raise ValueError("question is required")

    query_embedding = embedding_manager.embed_text(question)
    matches = vector_store.similarity_search(query_embedding, top_k=top_k, min_score=min_score, document_id=document_id)

    if not matches:
        return {"answer": "I don't have enough information from the uploaded documents to answer that.", "sources": []}

    answer = generate_answer(question, matches)

    sources = [{
        "filename": m["metadata"]["filename"],
        "chunkIndex": m["metadata"]["chunkIndex"],
        "score": round(m["score"], 4),
        "excerpt": m["text"][:240],
    } for m in matches]

    return {"answer": answer, "sources": sources}
