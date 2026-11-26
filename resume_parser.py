"""Extract plain text from uploaded resume files (PDF, DOCX, TXT)."""

from io import BytesIO

from docx import Document
from pypdf import PdfReader


class ResumeParseError(Exception):
    """Raised when a resume file can't be turned into usable text."""


MIN_WORDS = 40


def extract_text(filename: str, data: bytes) -> str:
    """Return the text content of a resume file based on its extension."""
    name = filename.lower()

    if name.endswith(".pdf"):
        text = _from_pdf(data)
    elif name.endswith(".docx"):
        text = _from_docx(data)
    elif name.endswith(".txt"):
        text = data.decode("utf-8", errors="ignore")
    else:
        raise ResumeParseError("Unsupported file type. Upload a PDF, DOCX or TXT file.")

    text = _clean(text)
    if len(text.split()) < MIN_WORDS:
        raise ResumeParseError(
            "Couldn't find enough text in that file. If it's a scanned image, "
            "paste the resume text instead."
        )
    return text


def _from_pdf(data: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:  # pypdf raises several error types
        raise ResumeParseError("That PDF couldn't be read. Try another file.") from exc


def _from_docx(data: bytes) -> str:
    try:
        doc = Document(BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as exc:
        raise ResumeParseError("That Word file couldn't be read. Try another file.") from exc


def _clean(text: str) -> str:
    """Normalize whitespace and drop empty lines."""
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)
