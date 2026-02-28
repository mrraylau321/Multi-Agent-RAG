import os
import json
import numpy as np
import faiss
import bm25s
from tqdm import tqdm
import ollama
import gensim.downloader as api
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import torch
from transformers import AutoTokenizer, AutoModel

_nltk_stopwords_ready = True
_nltk_punkt_ready = True
try:
    stopwords.words('english')
except LookupError:
    _nltk_stopwords_ready = False
try:
    word_tokenize("test")
except LookupError:
    _nltk_punkt_ready = False

def load_documents(file_path=None):
    if file_path is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        file_path = os.path.join(project_root, 'data', 'collection.jsonl')
        
    documents = []
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc="Loading documents from collection.jsonl"):
            documents.append(json.loads(line))
    return documents

class GuiOllamaDenseRetriever:
    def __init__(self, documents, model_name='mxbai-embed-large:335m', index_path=None, metadata_path=None):
        self.model_name = model_name
        self.client = ollama.Client(host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
        self.documents = documents
        self.doc_id_map = {doc['id']: doc for doc in self.documents}
        
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        if index_path and metadata_path:
            self.index_path = index_path
            self.metadata_path = metadata_path
        else:
            model_file_map = {
                "bge-m3:latest": ("faiss_index_bge-m3.bin", "chunk_metadata_bge-m3.json"),
                "qwen3-embedding:latest": ("faiss_index_qwen3-emb.bin", "chunk_metadata_qwen3-emb.json"),
            }
            index_file, metadata_file = model_file_map.get(model_name, ("faiss_index.bin", "chunk_metadata.json"))

            self.index_path = os.path.join(project_root, 'data', index_file)
            self.metadata_path = os.path.join(project_root, 'data', metadata_file)

        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'r') as f:
                self.chunk_metadata = json.load(f)
        else:
            raise FileNotFoundError(f"Index for {model_name} not found. Please build it first.")

    def _get_embedding(self, text):
        return self.client.embeddings(model=self.model_name, prompt=text)['embedding']

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        query_embedding = np.array([self._get_embedding(query)], dtype='float32')
        _, indices = self.index.search(query_embedding, top_k * 2)
        
        retrieved_doc_ids = [self.chunk_metadata[i]['doc_id'] for i in indices[0] if i < len(self.chunk_metadata)]
        return list(dict.fromkeys(retrieved_doc_ids))[:top_k]

    def get_docs_by_ids(self, doc_ids: list[str]) -> list[dict]:
        return [self.doc_id_map[doc_id] for doc_id in doc_ids if doc_id in self.doc_id_map]

class ColBERTRetriever:
    def __init__(self, documents, model_name='colbert-ir/colbertv2.0'):
        self.model_name = model_name
        self.documents = documents
        self.doc_id_map = {doc['id']: doc for doc in documents}
        
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.index_path = os.path.join(project_root, 'data', 'faiss_index_colbert.bin')
        self.metadata_path = os.path.join(project_root, 'data', 'chunk_metadata_colbert.json')

        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'r') as f:
                self.chunk_metadata = json.load(f)
        else:
            raise FileNotFoundError(f"ColBERT index not found at {self.index_path}. Please run the build script.")

    def _get_embedding(self, text: str) -> np.ndarray:
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
            outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return cls_embedding

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        query_embedding = self._get_embedding(query).astype('float32')
        _, indices = self.index.search(query_embedding, top_k * 2)
        
        retrieved_doc_ids = [self.chunk_metadata[i]['doc_id'] for i in indices[0] if i < len(self.chunk_metadata)]
        return list(dict.fromkeys(retrieved_doc_ids))[:top_k]

    def get_docs_by_ids(self, doc_ids: list[str]) -> list[dict]:
        return [self.doc_id_map[doc_id] for doc_id in doc_ids if doc_id in self.doc_id_map]

class StaticEmbeddingRetriever:
    def __init__(self, documents, model_name='glove-wiki-gigaword-100'):
        self.model_name = model_name
        self.documents = documents
        self.doc_id_map = {doc['id']: doc for doc in self.documents}
        self.model = api.load(self.model_name)
        self.embedding_dim = self.model.vector_size
        self.stop_words = set(stopwords.words('english')) if _nltk_stopwords_ready else set()
        
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.index_path = os.path.join(project_root, 'data', f'faiss_index_{self.model_name}.bin')

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            self.doc_ids = list(self.doc_id_map.keys())
        else:
            self._build_index()

    def _preprocess_text(self, text):
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Fallback to simple whitespace split when punkt is unavailable.
        tokens = word_tokenize(text) if _nltk_punkt_ready else text.split()
        return [w for w in tokens if w.isalpha() and w not in self.stop_words]

    def _text_to_vector(self, text):
        tokens = self._preprocess_text(text)
        vectors = [self.model[word] for word in tokens if word in self.model]
        if not vectors:
            return np.zeros(self.embedding_dim)
        return np.mean(vectors, axis=0)

    def _build_index(self):
        self.doc_ids = list(self.doc_id_map.keys())
        doc_embeddings = np.array([self._text_to_vector(self.doc_id_map[doc_id]['text']) for doc_id in tqdm(self.doc_ids, desc="Embedding documents")]).astype('float32')
        
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(doc_embeddings)
        
        faiss.write_index(self.index, self.index_path)

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        query_vector = np.array([self._text_to_vector(query)]).astype('float32')
        _, indices = self.index.search(query_vector, top_k)
        
        return [self.doc_ids[i] for i in indices[0]]

    def get_docs_by_ids(self, doc_ids: list[str]) -> list[dict]:
        return [self.doc_id_map[doc_id] for doc_id in doc_ids if doc_id in self.doc_id_map]

class BM25Retriever:
    def __init__(self, documents):
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

    def get_docs_by_ids(self, doc_ids: list[str]) -> list[dict]:
        return [self.doc_id_map[doc_id] for doc_id in doc_ids if doc_id in self.doc_id_map]

class GuiHybridRetriever:
    def __init__(self, documents, dense_model_name='mxbai-embed-large:335m'):
        self.documents = documents
        self.doc_id_map = {doc['id']: doc for doc in self.documents}
        self.bm25_retriever = BM25Retriever(documents)
        self.dense_retriever = GuiOllamaDenseRetriever(documents, model_name=dense_model_name)

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
