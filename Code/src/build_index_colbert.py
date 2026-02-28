import os
import sys
import json
import torch
import faiss
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# --- Path Setup ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.retrieval import load_documents

# --- Configuration ---
MODEL_NAME = 'colbert-ir/colbertv2.0'
OUTPUT_INDEX_PATH = os.path.join(PROJECT_ROOT, 'data', 'faiss_index_colbert.bin')
OUTPUT_METADATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'chunk_metadata_colbert.json')

# Chunking parameters
CHUNK_SIZE = 128  # Number of tokens per chunk
CHUNK_OVERLAP = 32 # Number of tokens to overlap between chunks
BATCH_SIZE = 64 # Number of chunks to process at once

def build_colbert_index():
    """
    Builds a FAISS index using single-vector representations ([CLS] token)
    from the ColBERTv2 model.
    """
    print("--- Starting ColBERTv2 Indexing Process ---")
    
    # 1. Load Documents
    print("Loading documents from collection.jsonl...")
    documents = load_documents()
    
    # 2. Load ColBERT Model and Tokenizer
    print(f"Loading model and tokenizer: {MODEL_NAME}...")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    embedding_dim = model.config.hidden_size

    # 3. Chunk Documents
    print(f"Chunking {len(documents)} documents... (Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})")
    chunk_metadata = []
    all_chunk_texts = []

    for doc in tqdm(documents, desc="Chunking documents"):
        doc_id = doc['id']
        text = doc['text']
        
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        for i in range(0, len(tokens), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_tokens = tokens[i : i + CHUNK_SIZE]
            if not chunk_tokens:
                continue
            
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            chunk_metadata.append({
                'doc_id': doc_id,
                'chunk_text': chunk_text
            })
            all_chunk_texts.append(chunk_text)

    print(f"Created {len(all_chunk_texts)} chunks.")

    # 4. Generate Embeddings in Batches
    print("Generating embeddings for all chunks...")
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(all_chunk_texts), BATCH_SIZE), desc="Embedding chunks"):
            batch_texts = all_chunk_texts[i : i + BATCH_SIZE]
            
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=CHUNK_SIZE + 2 # Account for [CLS] and [SEP]
            ).to(device)
            
            outputs = model(**inputs)
            
            # Use the [CLS] token's embedding as the representation for the chunk
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)
    
    embeddings_matrix = np.vstack(all_embeddings).astype('float32')
    print(f"Generated embeddings matrix of shape: {embeddings_matrix.shape}")

    # 5. Build and Save FAISS Index
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_matrix)
    
    print(f"Saving FAISS index to {OUTPUT_INDEX_PATH}...")
    faiss.write_index(index, OUTPUT_INDEX_PATH)

    # 6. Save Metadata
    print(f"Saving chunk metadata to {OUTPUT_METADATA_PATH}...")
    with open(OUTPUT_METADATA_PATH, 'w') as f:
        json.dump(chunk_metadata, f)

    print("--- Indexing Complete ---")
    print(f"Total chunks / vectors in index: {index.ntotal}")

if __name__ == "__main__":
    build_colbert_index()
