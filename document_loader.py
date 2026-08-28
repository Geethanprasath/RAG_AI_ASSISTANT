from pathlib import Path
from pypdf import PdfReader
from docx import Document


def load_file(file_path):
    path = Path(file_path)

    if path.suffix.lower() == ".pdf":
        return load_pdf(path)

    elif path.suffix.lower() == ".docx":
        return load_docx(path)

    elif path.suffix.lower() == ".txt":
        return load_txt(path)

    else:
        raise ValueError("Unsupported file type")


def load_pdf(path):
    reader = PdfReader(path)

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return pages


def load_docx(path):
    document = Document(path)

    text = "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )

    return [text]


def load_txt(path):
    text = path.read_text(encoding="utf-8")

    return [text]