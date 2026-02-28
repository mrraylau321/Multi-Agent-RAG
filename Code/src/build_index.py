import os
import json
import faiss
import numpy as np
import ollama
from tqdm import tqdm
from transformers import AutoTokenizer
import re
import time
import random

# --- Configuration ---
MODEL_NAME = 'mxbai-embed-large:335m'
TOKENIZER_NAME = 'mixedbread-ai/mxbai-embed-large-v1'
COLLECTION_PATH = 'data/collection.jsonl'
INDEX_SAVE_PATH = 'data/faiss_index.bin'
CHUNK_METADATA_SAVE_PATH = 'data/chunk_metadata.json'
CHUNK_SIZE = 480  # Max tokens per chunk
CHUNK_OVERLAP = 50 # Tokens to overlap between chunks
MODEL_CONTEXT_LIMIT = 512 # The official context limit for the model
# ---------------------

# Configure Ollama client
ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
print(f"Connecting to Ollama at: {ollama_host}")
client = ollama.Client(host=ollama_host)

# Load Tokenizer
print(f"Loading tokenizer: {TOKENIZER_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
print("Tokenizer loaded successfully.")

def get_embedding_dim():
    """Gets the embedding dimension from the Ollama model."""
    try:
        response = client.embeddings(model=MODEL_NAME, prompt='.')
        return len(response['embedding'])
    except Exception as e:
        print(f"Could not determine embedding dimension from Ollama. Error: {e}")
        print("Please ensure Ollama is running and the model is available.")
        print("Falling back to dimension 1024.")
        return 1024

def split_text_into_chunks(text, chunk_size, chunk_overlap):
    """
    Splits text into overlapping chunks based on token count.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        return []

    chunks = []
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = start_idx + chunk_size
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        # Move the start index for the next chunk, considering overlap
        start_idx += chunk_size - chunk_overlap
        
        # If we're at the end and the last chunk is just overlap, stop
        if start_idx >= len(tokens) and len(tokens) - (start_idx - (chunk_size - chunk_overlap)) < chunk_overlap:
            break
            
    return chunks

def embed_with_retries(text: str, client: ollama.Client, model_name: str, max_retries: int = 6, base_delay: float = 0.5):
    """
    Call Ollama embeddings with robust retries to tolerate transient EOF/500 errors.
    """
    attempt = 0
    while True:
        try:
            return client.embeddings(model=model_name, prompt=text)
        except Exception as e:
            attempt += 1
            if attempt >= max_retries:
                print(f"\nError: embedding failed after {attempt} attempts: {e}")
                return None
            delay = base_delay * (2 ** (attempt - 1)) * (0.8 + 0.4 * random.random())
            print(f"\nWarning: embedding error (attempt {attempt}/{max_retries}): {e} | retrying in {delay:.2f}s")
            time.sleep(delay)

def build_and_save_index():
    """
    Loads the document collection, splits documents into overlapping chunks, 
    generates an embedding for each chunk, builds a FAISS index, and saves it.
    """
    print("--- Starting FAISS Index Build Process (Overlapping Multi-Vector Chunking) ---")

    # 1. Load Documents
    print(f"Loading documents from {COLLECTION_PATH}...")
    try:
        with open(COLLECTION_PATH, 'r') as f:
            documents = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: Document collection not found at \'{COLLECTION_PATH}\'.")
        print("Please run the download script first.")
        return
    print(f"Loaded {len(documents)} documents.")

    # 2. Create Chunks and Generate Embeddings
    print(f"Generating embeddings for document chunks using Ollama model: {MODEL_NAME}")
    print(f"Chunk Size: {CHUNK_SIZE} tokens, Overlap: {CHUNK_OVERLAP} tokens")
    print("This will take a significant amount of time.")

    embedding_dim = get_embedding_dim()
    all_embeddings = []
    chunk_metadata = []

    for doc in tqdm(documents, desc="Chunking and Encoding Documents"):
        text = doc.get('text', '')
        if not text.strip():
            continue

        # Use the new overlapping chunking strategy
        chunks = split_text_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
        
        for chunk in chunks:
            if not chunk.strip():
                continue

            # Final safety check: ensure the chunk is not too long for the model
            chunk_tokens = tokenizer.encode(chunk, add_special_tokens=False)
            if len(chunk_tokens) >= MODEL_CONTEXT_LIMIT:
                # Truncate the tokens and decode back to text
                chunk_tokens = chunk_tokens[:MODEL_CONTEXT_LIMIT - 1]
                chunk = tokenizer.decode(chunk_tokens)
                if not chunk.strip():
                    continue
            
            response = embed_with_retries(chunk, client, MODEL_NAME)
            if not response or 'embedding' not in response:
                print(f"\nWarning: Could not embed chunk for doc ID {doc['id']} after retries. Skipping chunk.")
                continue
            all_embeddings.append(response['embedding'])
            chunk_metadata.append({
                'doc_id': doc['id'],
                'chunk_text': chunk
            })

    if not all_embeddings:
        print("Error: No embeddings were generated. Cannot build index.")
        return

    embeddings_np = np.array(all_embeddings).astype('float32')
    
    # 3. Build and Save FAISS Index
    print(f"\nBuilding FAISS index on {embeddings_np.shape[0]} chunk vectors...")
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    print(f"FAISS index built successfully.")

    print(f"Saving FAISS index to {INDEX_SAVE_PATH}...")
    faiss.write_index(index, INDEX_SAVE_PATH)
    print("Index saved successfully.")

    # 4. Save Chunk Metadata
    print(f"Saving chunk metadata to {CHUNK_METADATA_SAVE_PATH}...")
    with open(CHUNK_METADATA_SAVE_PATH, 'w') as f:
        json.dump(chunk_metadata, f)
    print("Chunk metadata saved successfully.")

    print("\n--- Index Build Process Complete ---")

if __name__ == "__main__":
    build_and_save_index()