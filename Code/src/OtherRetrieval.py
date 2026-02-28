import os
import json
import re
import faiss
import numpy as np
import gensim.downloader as api
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Download NLTK data ---
try:
    stopwords.words('english')
except LookupError:
    print("NLTK stopwords not found. Downloading...")
    nltk.download('stopwords')
    nltk.download('punkt')
    print("NLTK data downloaded.")

class StaticEmbeddingRetriever:
    def __init__(self, documents, model_name='glove-wiki-gigaword-100', use_prebuilt_index=True):
        print("Initializing Static Embedding Retriever...")
        self.model_name = model_name
        self.documents = documents
        self.doc_id_map = {doc['id']: doc for doc in self.documents}
        
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.index_path = os.path.join(project_root, 'data', f'faiss_index_{self.model_name}.bin')
        
        print(f"Loading GloVe model: {self.model_name}. This may take some time...")
        self.model = api.load(self.model_name)
        self.embedding_dim = self.model.vector_size
        print("GloVe model loaded.")

        self.stop_words = set(stopwords.words('english'))

        if use_prebuilt_index and os.path.exists(self.index_path):
            print(f"Loading pre-built FAISS index for {self.model_name}...")
            self.index = faiss.read_index(self.index_path)
            self.doc_ids = list(self.doc_id_map.keys()) # Assuming order is preserved, which is risky.
                                                      # A better approach would be to save doc_ids with the index.
                                                      # For now, this will work if the document set is static.
            print(f"FAISS index with {self.index.ntotal} vectors loaded.")
        else:
            print("Building new FAISS index for static embeddings...")
            self._build_index()

    def _preprocess_text(self, text):
        text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = word_tokenize(text)
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
        
        print(f"Saving FAISS index to {self.index_path}...")
        faiss.write_index(self.index, self.index_path)
        print("Index saved.")

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        query_vector = np.array([self._text_to_vector(query)]).astype('float32')
        _, indices = self.index.search(query_vector, top_k)
        
        return [self.doc_ids[i] for i in indices[0]]

    def score(self, query: str, doc_ids: list[str]) -> dict[str, float]:
        query_vector = self._text_to_vector(query)
        scores = {}
        for doc_id in doc_ids:
            if doc_id in self.doc_id_map:
                doc_vector = self._text_to_vector(self.doc_id_map[doc_id]['text'])
                # Cosine similarity
                sim = np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
                scores[doc_id] = float(sim)
        return scores
