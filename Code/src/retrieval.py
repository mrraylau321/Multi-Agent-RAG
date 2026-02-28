import os
import json
import numpy as np
import faiss
import bm25s
from tqdm import tqdm
import ollama

# --- Dense Retriever using Ollama ---
class OllamaDenseRetriever:
    def __init__(self, documents, model_name='mxbai-embed-large:335m', use_prebuilt_index=True):
        print("Initializing Ollama Dense Retriever...")
        self.model_name = model_name
        ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.client = ollama.Client(host=ollama_host)
        print(f"Connecting to Ollama at: {ollama_host}")
        
        self.documents = documents
        self.doc_id_map = {doc['id']: doc for doc in self.documents}
        
        
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        # Map model names to specific file names
        model_file_map = {
            "bge-m3:latest": ("faiss_index_bge-m3.bin", "chunk_metadata_bge-m3.json"),
            "qwen3-embedding:latest": ("faiss_index_qwen3-emb.bin", "chunk_metadata_qwen3-emb.json"),
        }
        
        # Get the specific files for the model, or use the default
        index_file, metadata_file = model_file_map.get(model_name, ("faiss_index.bin", "chunk_metadata.json"))

        self.index_path = os.path.join(project_root, 'data', index_file)
        self.metadata_path = os.path.join(project_root, 'data', metadata_file)


        if use_prebuilt_index and os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            print("Loading pre-built FAISS index and chunk metadata...")
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'r') as f:
                self.chunk_metadata = json.load(f)
            print(f"FAISS index with {self.index.ntotal} vectors and metadata for {len(self.chunk_metadata)} chunks loaded successfully.")
        else:
            raise FileNotFoundError("Pre-built FAISS index not found. Please run the index building script.")

    def _get_embedding(self, text):
        return self.client.embeddings(model=self.model_name, prompt=text)['embedding']

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        query_embedding = np.array([self._get_embedding(query)], dtype='float32')
        _, indices = self.index.search(query_embedding, top_k * 2)
        
        retrieved_doc_ids = []
        for i in indices[0]:
            if i < len(self.chunk_metadata):
                retrieved_doc_ids.append(self.chunk_metadata[i]['doc_id'])
        
        return list(dict.fromkeys(retrieved_doc_ids))[:top_k]

    def score(self, query: str, doc_ids: list[str]) -> dict[str, float]:
        query_embedding = np.array(self._get_embedding(query), dtype='float32')
        scores = {}
        
        doc_id_to_positions = {}
        for i, md in enumerate(self.chunk_metadata):
            doc_id = md['doc_id']
            if doc_id in doc_ids:
                doc_id_to_positions.setdefault(doc_id, []).append(i)

        for doc_id, positions in doc_id_to_positions.items():
            doc_embeddings = np.array([self.index.reconstruct(pos) for pos in positions])
            sims = np.dot(doc_embeddings, query_embedding) / (np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding))
            scores[doc_id] = float(np.max(sims))
        return scores

# --- Sparse Retriever using BM25 ---
class BM25Retriever:
    def __init__(self, documents):
        print("Initializing BM25 Retriever...")
        self.documents = documents
        self.doc_ids = [doc['id'] for doc in self.documents]
        self.doc_id_map = {doc['id']: doc for doc in self.documents}
        
        corpus = [doc['text'] for doc in self.documents]
        self.tokenized_corpus = bm25s.tokenize(corpus)
        
        self.index = bm25s.BM25()
        self.index.index(self.tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        tokenized_query = bm25s.tokenize(query)
        results_idx, _ = self.index.retrieve(tokenized_query, k=top_k)
        return [self.doc_ids[i] for i in results_idx[0]]

    def score(self, query: str, doc_ids: list[str]) -> dict[str, float]:
        tokenized_query = bm25s.tokenize(query)
        
        docs_to_score_texts = []
        valid_doc_ids = []
        for doc_id in doc_ids:
            if doc_id in self.doc_id_map:
                docs_to_score_texts.append(self.doc_id_map[doc_id]['text'])
                valid_doc_ids.append(doc_id)

        if not valid_doc_ids:
            return {}

        # Tokenize only the subset of documents we need to score
        tokenized_docs_to_score = bm25s.tokenize(docs_to_score_texts)
        
        # The .score method calculates scores for a given query against a (potentially un-indexed) corpus
        scores_array = self.index.score(tokenized_query, doc_representations=tokenized_docs_to_score)
        
        return {valid_doc_ids[i]: float(scores_array[0][i]) for i in range(len(valid_doc_ids))}

# --- Hybrid Retriever ---
class HybridRetriever:
    def __init__(self, documents, dense_model_name='mxbai-embed-large:335m'):
        print("Initializing Hybrid Retriever...")
        self.documents = documents
        self.doc_id_map = {doc['id']: doc for doc in self.documents}
        self.bm25_retriever = BM25Retriever(documents)
        self.dense_retriever = OllamaDenseRetriever(documents, model_name=dense_model_name)
        print("Hybrid Retriever Initialized.")

    def retrieve(self, query: str, top_k: int = 10, alpha: float = 0.5):
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k)
        dense_results = self.dense_retriever.retrieve(query, top_k=top_k)
        
        scores = {}
        for rank, doc_id in enumerate(bm25_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rank + 10)
        for rank, doc_id in enumerate(dense_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rank + 10)
            
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs[:top_k]]

    def get_docs_by_ids(self, doc_ids: list[str]) -> list[dict]:
        return [self.doc_id_map[doc_id] for doc_id in doc_ids if doc_id in self.doc_id_map]

def load_documents(file_path=None):
    if file_path is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        file_path = os.path.join(project_root, 'data', 'collection.jsonl')
        
    documents = []
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc="Loading documents from collection.jsonl"):
            documents.append(json.loads(line))
    return documents