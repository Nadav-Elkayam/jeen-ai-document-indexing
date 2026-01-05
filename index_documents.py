import os
import uuid
from datetime import datetime
from typing import List
from dotenv import load_dotenv
load_dotenv()
import psycopg2
from psycopg2.extras import execute_values
import google.generativeai as genai
from pypdf import PdfReader
from docx import Document

# Config
# PostgreSQL connection settings (defaults match your Docker setup)
DB_HOST = os.getenv("PGHOST", "localhost")
DB_PORT = int(os.getenv("PGPORT", "5434"))  # Docker mapping: localhost:5434 -> container:5432
DB_NAME = os.getenv("PGDATABASE", "jeen_ai")
DB_USER = os.getenv("PGUSER", "postgres")
DB_PASSWORD = os.getenv("PGPASSWORD", "postgres")

# Gemini API settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = "models/text-embedding-004"

# Chunking strategy
SPLIT_STRATEGY = "fixed_overlap"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# File reading
# Read a PDF file and return all extracted text as a single string.
def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    parts: List[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()

#Read a DOCX file and return its paragraph text as a single string.
def read_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs).strip()


#Load text from a supported file type: .PDF , .docx
def load_text(path: str) -> str:
    p = path.lower()
    if p.endswith(".pdf"):
        return read_pdf(path)
    if p.endswith(".docx"):
        return read_docx(path)
    raise ValueError("Unsupported file type. Please provide a PDF or DOCX file.")


#Make the text easier to chunk by collapsing all whitespace into single spaces.
def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())

#Read a PDF file and return all extracted text as a single string.
def split_fixed_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    text = normalize_whitespace(text)
    if not text:
        return []

    chunks: List[str] = []
    step = chunk_size - overlap

    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)

    return chunks


#Create embeddings for a list of text chunks using Gemini.
#Returns: A list of vectors (each vector is a list[float]) in the same order as `texts`.
def embed_texts(texts: List[str]) -> List[List[float]]:
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY env var. Set it before running.")

    genai.configure(api_key=GEMINI_API_KEY)

    vectors: List[List[float]] = []
    for t in texts:
        resp = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=t,
        )
        vec = resp["embedding"]
        vectors.append(vec)

    return vectors

#Open a PostgreSQL connection using env vars / defaults above.
def db_connect():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )

#Insert chunk text + embeddings into the `document_chunks` table.
def insert_chunks(filename: str, chunks: List[str], embeddings: List[List[float]]):
    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings length mismatch.")

    now = datetime.utcnow()
    rows = [
        (
            str(uuid.uuid4()),
            chunk_text,
            emb,               # pgvector מקבל array בפורמט [..]
            filename,
            SPLIT_STRATEGY,
            now,
        )
        for chunk_text, emb in zip(chunks, embeddings)
    ]

    sql = """
        INSERT INTO document_chunks
        (id, chunk_text, embedding, filename, split_strategy, created_at)
        VALUES %s
    """

    with db_connect() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows)
        conn.commit()



def main():
    import argparse

    parser = argparse.ArgumentParser(description="Index PDF/DOCX into PostgreSQL (pgvector).")
    parser.add_argument("path", help="Path to a PDF or DOCX file to index")
    args = parser.parse_args()

    path = args.path
    filename = os.path.basename(path)

    text = load_text(path)
    if not text:
        print("No text extracted from the file. Nothing to index.")
        return

    chunks = split_fixed_overlap(text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Extracted {len(text)} chars → {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    embeddings = embed_texts(chunks)

    dims = len(embeddings[0]) if embeddings else 0
    print(f"Created {len(embeddings)} embeddings. dims={dims}")

    insert_chunks(filename, chunks, embeddings)
    print("Inserted into document_chunks successfully.")


if __name__ == "__main__":
    main()
