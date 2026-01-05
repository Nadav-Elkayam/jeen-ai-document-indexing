# Jeen AI – Document Indexing with Embeddings

This project indexes PDF and DOCX documents into a PostgreSQL database using vector embeddings.
Each document is split into overlapping text chunks, embedded using Gemini, and stored with pgvector
for later retrieval or AI-based analysis.

The goal is to demonstrate a clean and practical document processing pipeline.

---

## What This Project Does

1. Reads a PDF or DOCX file  
2. Extracts the raw text  
3. Splits the text into fixed-size chunks with overlap  
4. Creates embeddings for each chunk using Gemini  
5. Stores chunks and embeddings in PostgreSQL (pgvector)

---

## Chosen Strategy – Fixed Overlap Chunking

This project uses a **fixed window chunking strategy with overlap**.

**Why this strategy?**
- Simple and reliable  
- Preserves context between chunks  
- Works well as a baseline for embedding-based systems  

**How it works:**
- Each chunk has a fixed size (default: 1000 characters)  
- Consecutive chunks overlap by a fixed amount (default: 200 characters)  
- Overlap helps prevent losing meaning at chunk boundaries  

---

## Installation

1. Clone the repository  
2. Create a virtual environment  
3. Install the required dependencies  

```bash
pip install -r requirements.txt
```

## Environment Variables
Create a .env file in the project root:
```
PGHOST=localhost
PGPORT=5434
PGDATABASE=jeen_ai
PGUSER=postgres
PGPASSWORD=postgres
GEMINI_API_KEY=your_api_key_here
```
Usage Example
The project was tested using the following files:
sample_test_document.pdf
sample_test_document.docx

## Run the script using:
```
python index_documents.py sample_test_document.pdf
```
or

```
python index_documents.py sample_test_document.docx
```
## Example output:
```
Extracted 12487 chars → 16 chunks.
Created 16 embeddings.
Inserted into document_chunks successfully.
```

