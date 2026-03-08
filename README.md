# Local RAG Knowledge Assistant

A simple Retrieval-Augmented Generation (RAG) assistant for document-based question answering.

The system retrieves relevant document chunks using vector search and generates answers with a local LLM.

## Tech Stack

- Python
- FAISS
- Sentence Transformers
- Ollama
- Qwen LLM
- Gradio

## Features

- Upload TXT / Markdown documents
- Retrieve relevant context with FAISS
- Generate answers with a local LLM
- Interactive chat interface

## Adjustable Parameters

You may need to modify the following parameters in `rag_pipeline.py`.

### LLM model
LLM_MODEL = "qwen3:4b"


Change this depending on the model installed in Ollama.

Example:
ollama pull qwen3:4b


---

### Top-K retrieval
TOP_K = 3


Controls how many document chunks are retrieved for answering.

Typical values:
3 - fast
5 - more context


---

### Embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


You can replace it with other SentenceTransformer models.

---

## Run the Project

Install dependencies:
pip install -r requirements.txt


Run the application:
python app.py


Then open the web interface.

---

## Architecture

1. Document ingestion
2. Text chunking
3. Embedding generation
4. Vector retrieval
5. LLM answer generation
