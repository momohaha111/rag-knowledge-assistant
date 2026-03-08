# Local RAG Knowledge Assistant

A local Retrieval-Augmented Generation (RAG) assistant for document-based question answering.

## Features

- Upload TXT or Markdown documents
- Retrieve relevant context using FAISS vector search
- Generate answers using a local LLM via Ollama
- Interactive chat interface built with Gradio

## Tech Stack

- Python
- FAISS
- Sentence Transformers
- Ollama
- Qwen LLM
- Gradio

## Architecture

1. Document ingestion
2. Text chunking
3. Embedding generation
4. Vector retrieval
5. LLM response generation

## Run Locally

```bash
pip install -r requirements.txt
python app.py
