"""Groq chat completion grounded in retrieved context (port of llmService.js)."""
from config import config

_client = None

def _get_client():
    global _client
    if not config.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to .env")
    if _client is None:
        from groq import Groq
        _client = Groq(api_key=config.groq_api_key)
    return _client

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context, extracted from documents the user uploaded.
- If the answer is not contained in the context, say you don't have enough information from the uploaded documents.
- Be concise and accurate.
- Do not make up information that is not in the context."""

def generate_answer(question, context_chunks):
    groq = _get_client()

    context = "\n\n---\n\n".join(
        f"[Source {i + 1} - {c['metadata']['filename']}, chunk {c['metadata']['chunkIndex']}]\n{c['text']}"
        for i, c in enumerate(context_chunks)
    )

    user_prompt = f"Context:\n{context or '(no relevant context found)'}\n\nQuestion: {question}\n\nAnswer based only on the context above."

    completion = groq.chat.completions.create(
        model=config.groq_model,
        temperature=0.1,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return ((completion.choices[0].message.content if completion.choices else "") or "").strip()
