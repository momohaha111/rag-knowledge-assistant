import faiss
import ollama
from sentence_transformers import SentenceTransformer

# ---------------- Config ----------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "qwen3:4b"
TOP_K = 3

# Load embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

VECTOR_DIM = 384
index = faiss.IndexFlatL2(VECTOR_DIM)

text_chunks = []

# ---------------- Functions ----------------

def load_document(file_path):

    global index, text_chunks

    index.reset()
    text_chunks = []

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    text_chunks = [c.strip() for c in text.split("\n") if c.strip()]

    if len(text_chunks) == 0:
        return "Document is empty."

    embeddings = embedding_model.encode(text_chunks)

    index.add(embeddings)

    return f"Loaded {len(text_chunks)} chunks into vector store."


def retrieve_context(query):

    query_embedding = embedding_model.encode([query])

    distances, indices = index.search(query_embedding, k=TOP_K)

    context = "\n".join(
        [text_chunks[i] for i in indices[0] if i >= 0]
    )

    return context


def generate_answer(query):

    context = retrieve_context(query)

    prompt = f"""
Answer the user question strictly based on the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]
