"""PDF text extraction (port of pdfService.js; pdf-parse -> pypdf)."""
from pypdf import PdfReader

def extract_text_from_pdf(file_path):
    reader = PdfReader(str(file_path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return {
        "text": "\n\n".join(pages),
        "numPages": len(reader.pages) or 0,
        "info": dict(reader.metadata or {}),
    }
