"""RAG PDF Chat - Streamlit UI (replaces the React frontend + Express routes; pipeline logic lives in services/)."""
import uuid
from pathlib import Path
import streamlit as st

from config import config
from services.rag_service import ingest_multiple_pdfs, ask_question
from services.vector_store import vector_store
from services.document_registry import document_registry

config.uploads_dir.mkdir(parents=True, exist_ok=True)
config.vector_store_dir.mkdir(parents=True, exist_ok=True)

STATUS_LABEL = {
    "processing": "Processing…",
    "ready": "Ready",
    "empty": "No extractable text",
    "failed": "Failed",
}

st.set_page_config(page_title="RAG PDF Chat", page_icon="📄", layout="wide")

# ---- session state --------------------------------------------------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("upload_error", None)

# ---- header ---------------------------------------------------------------
st.title("RAG PDF Chat")
st.caption("Upload PDFs, then ask questions grounded in their content.")


# ---- upload (POST /api/upload) -------------------------------------------
def handle_upload(files):
    """Same rules as upload.middleware.js: PDFs only, max 20 files, 25 MB each."""
    pdfs = [f for f in files if f.type == "application/pdf"]
    if not pdfs:
        return "Please select PDF files only."
    if len(pdfs) > config.max_files:
        return f"Too many files. Maximum is {config.max_files}."
    too_big = [f.name for f in pdfs if f.size > config.max_file_size]
    if too_big:
        return f"File too large (max 25 MB): {', '.join(too_big)}"

    saved = []
    for f in pdfs:
        dest = config.uploads_dir / f"{uuid.uuid4()}{Path(f.name).suffix}"
        dest.write_bytes(f.getbuffer())
        saved.append((dest, f.name))
    try:
        with st.spinner(f"Processing {len(saved)} file(s) — chunking & embedding…"):
            ingest_multiple_pdfs(saved)
    except Exception as err:  # errorHandler equivalent
        return str(err)
    return None


with st.sidebar:
    files = st.file_uploader(
        "Drag & drop PDFs here, or click to browse",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can select multiple files at once",
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if files:
        st.session_state.upload_error = handle_upload(files)
        st.session_state.uploader_key += 1  # clear the picker, like inputRef.value = ""
        st.rerun()
    if st.session_state.upload_error:
        st.error(st.session_state.upload_error)

    st.subheader("Documents")
    documents = document_registry.list()
    if not documents:
        st.caption("No documents uploaded yet.")
    for doc in documents:
        pages, chunks = doc.get("numPages", 0), doc.get("numChunks", 0)
        st.markdown(f"**{doc['filename']}**")
        st.caption(
            f"{pages} page{'' if pages == 1 else 's'} · {chunks} chunk{'' if chunks == 1 else 's'} · "
            f"{STATUS_LABEL.get(doc['status'], doc['status'])}"
        )
        if st.button("Remove", key=f"del_{doc['id']}"):
            # DELETE /api/documents/:id  -> remove vector chunks + metadata
            vector_store.remove_by_document_id(doc["id"])
            document_registry.remove(doc["id"])
            st.rerun()

ready_count = sum(1 for d in document_registry.list() if d["status"] == "ready")
disabled = ready_count == 0


# ---- chat (POST /api/query) ----------------------------------------------
def render_message(m):
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("sources"):
            st.markdown("**Sources**")
            for s in m["sources"]:
                st.markdown(f"- **{s['filename']}** · chunk {s['chunkIndex']} · score {s['score']}")
                st.caption(f"*{s['excerpt']}…*")


if not st.session_state.messages:
    st.info("Upload a PDF to start asking questions." if disabled else "Ask a question about your uploaded documents.")

for m in st.session_state.messages:
    render_message(m)

question = st.chat_input(
    "Upload a PDF first…" if disabled else "Ask a question about your documents…", disabled=disabled
)
if question and question.strip():
    question = question.strip()
    st.session_state.messages.append({"role": "user", "content": question})
    render_message(st.session_state.messages[-1])
    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = ask_question(question)
        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
        )
    except Exception as err:
        st.session_state.messages.append({"role": "assistant", "content": f"Error: {err}"})
    st.rerun()
