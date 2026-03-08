import faiss
import ollama
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
LLM_MODEL = 'qwen3:4b'
VECTOR_DIM = 384

index = faiss.IndexFlatL2(VECTOR_DIM)
text_chunks = []

def load_document(file_path):
    global index, text_chunks
    index.reset()
    text_chunks = []

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    text_chunks = [chunk.strip() for chunk in text.split('\n') if chunk.strip()]

    embeddings = EMBEDDING_MODEL.encode(text_chunks)
    index.add(embeddings)

    return f"Loaded {len(text_chunks)} text chunks."

def rag_chat(message):
    query_embedding = EMBEDDING_MODEL.encode([message])
    distances, indices = index.search(query_embedding, k=3)

    context = "\n".join([text_chunks[idx] for idx in indices[0]])

    prompt = f"""
Answer the question strictly based on the context.

Context:
{context}

Question:
{message}

Answer:
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content']
