import json
import re


def clean_text(text):
    """Clean unnecessary whitespace from text."""

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks."""

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def save_chunks(chunks, filename="data/chunks.json"):
    """Save chunks to JSON file."""

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=4, ensure_ascii=False)