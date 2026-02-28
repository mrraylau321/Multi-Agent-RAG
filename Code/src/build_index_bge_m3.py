import os
import json
import faiss
import numpy as np
import ollama
from tqdm import tqdm
from transformers import AutoTokenizer
import time
import random

# --- Configuration ---
MODEL_NAME = 'bge-m3'  # Ollama model name
TOKENIZER_NAME = 'BAAI/bge-m3'
COLLECTION_PATH = 'data/collection.jsonl'
INDEX_SAVE_PATH = 'data/faiss_index_bge-m3.bin'
CHUNK_METADATA_SAVE_PATH = 'data/chunk_metadata_bge-m3.json'
CHUNK_SIZE = 768          # tokens per chunk
CHUNK_OVERLAP = 128       # token overlap between chunks
MODEL_CONTEXT_LIMIT = 8192  # conservative cap for chunk safety
# ---------------------

# Configure Ollama client
ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
print(f"Connecting to Ollama at: {ollama_host}")
client = ollama.Client(host=ollama_host)

print(f"Loading tokenizer: {TOKENIZER_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
print("Tokenizer loaded successfully.")


def get_embedding_dim() -> int:
    """Determine embedding dimension by probing a trivial prompt."""
    try:
        response = client.embeddings(model=MODEL_NAME, prompt=".")
        return len(response['embedding'])
    except Exception as e:
        print(f"Warning: Could not determine embedding dim from Ollama: {e}")
        print("Falling back to dimension 1024.")
        return 1024


def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int):
    """
    Split text into overlapping token-based chunks using the HF tokenizer.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        return []
    chunks = []
    start_idx = 0
    step = max(1, chunk_size - chunk_overlap)
    while start_idx < len(tokens):
        end_idx = start_idx + chunk_size
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens)
        if chunk_text:
            chunks.append(chunk_text)
        start_idx += step
        # Stop if remaining is just overlap-sized tail
        if start_idx >= len(tokens) and len(tokens) - (start_idx - step) < chunk_overlap:
            break
    return chunks


def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    """
    L2-normalize each row vector in-place-safe manner and return the normalized matrix.
    """
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return mat / norms


def embed_with_retries(text: str, max_retries: int = 6, base_delay: float = 0.5):
    """
    Call Ollama embeddings with robust retries to tolerate transient EOF/500 errors.
    """
    attempt = 0
    while True:
        try:
            return client.embeddings(model=MODEL_NAME, prompt=text)
        except Exception as e:
            attempt += 1
            if attempt >= max_retries:
                print(f"\nError: embedding failed after {attempt} attempts: {e}")
                return None
            # Exponential backoff with jitter
            delay = base_delay * (2 ** (attempt - 1)) * (0.8 + 0.4 * random.random())
            print(f"\nWarning: embedding error (attempt {attempt}/{max_retries}): {e} | retrying in {delay:.2f}s")
            time.sleep(delay)


def build_and_save_index():
    """
    Builds a cosine-similarity FAISS index (IP on normalized vectors) from collection.jsonl
    using Ollama bge-m3 embeddings and overlapping chunks.
    """
    print("--- Building FAISS index with bge-m3 (cosine via IP on normalized vectors) ---")

    # 1) Load documents
    print(f"Loading documents from {COLLECTION_PATH}...")
    try:
        with open(COLLECTION_PATH, 'r') as f:
            documents = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: Document collection not found at '{COLLECTION_PATH}'.")
        print("Please run the dataset download first.")
        return
    print(f"Loaded {len(documents)} documents.")

    # 2) Chunking + Embedding
    print(f"Generating embeddings via Ollama model: {MODEL_NAME}")
    print(f"Chunk Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    print("This may take a while depending on dataset size and hardware.")

    embedding_dim = get_embedding_dim()
    all_embeddings: list[list[float]] = []
    chunk_metadata: list[dict] = []

    for doc in tqdm(documents, desc="Chunking and Encoding Documents"):
        text = doc.get('text', '')
        if not text or not isinstance(text, str):
            continue

        chunks = split_text_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk in chunks:
            if not chunk:
                continue
            # Safety: ensure chunk under model context limit
            tok = tokenizer.encode(chunk, add_special_tokens=False)
            if len(tok) >= MODEL_CONTEXT_LIMIT:
                tok = tok[:MODEL_CONTEXT_LIMIT - 1]
                chunk = tokenizer.decode(tok)
                if not chunk:
                    continue
            resp = embed_with_retries(chunk)
            if not resp or 'embedding' not in resp:
                print(f"\nWarning: embedding failed for doc_id={doc.get('id')} after retries; skipping chunk.")
                continue
            emb = resp['embedding']
            if not isinstance(emb, list) or not emb:
                continue
            all_embeddings.append(emb)
            chunk_metadata.append({
                'doc_id': doc['id'],
                'chunk_text': chunk
            })

    if not all_embeddings:
        print("Error: No embeddings generated; cannot build index.")
        return

    emb_np = np.array(all_embeddings, dtype='float32')
    if emb_np.shape[1] != embedding_dim:
        print(f"Note: probed dim {embedding_dim}, actual {emb_np.shape[1]}; proceeding with actual.")

    # L2-normalize for cosine similarity with IP index
    emb_np = l2_normalize_rows(emb_np)

    # 3) Build FAISS Index (IP)
    print(f"\nBuilding FAISS IndexFlatIP on {emb_np.shape[0]} chunk vectors (dim={emb_np.shape[1]})...")
    index = faiss.IndexFlatIP(emb_np.shape[1])
    index.add(emb_np)
    print("FAISS index constructed.")

    # 4) Save index and metadata
    os.makedirs(os.path.dirname(INDEX_SAVE_PATH), exist_ok=True)
    print(f"Saving FAISS index to {INDEX_SAVE_PATH} ...")
    faiss.write_index(index, INDEX_SAVE_PATH)
    print("Index saved.")

    print(f"Saving chunk metadata to {CHUNK_METADATA_SAVE_PATH} ...")
    with open(CHUNK_METADATA_SAVE_PATH, 'w') as f:
        json.dump(chunk_metadata, f)
    print("Chunk metadata saved.")

    print("\n--- Completed building bge-m3 index ---")


if __name__ == "__main__":
    build_and_save_index()


