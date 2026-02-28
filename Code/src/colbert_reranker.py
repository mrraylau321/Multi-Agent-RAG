import os
import torch
import numpy as np
from typing import List, Tuple, Dict
from transformers import AutoTokenizer, AutoModel


class ColBERTReranker:
    """
    Lightweight ColBERT reranker (zero-shot) that reorders candidate documents
    returned by your existing retriever. It does NOT build a standalone index.

    Scoring: ColBERT MaxSim
      score(q, d) = sum_i max_j dot( q_i / ||q_i||, d_j / ||d_j|| )
    """

    def __init__(self,
                 model_name: str = "colbert-ir/colbertv2.0",
                 max_query_tokens: int = 32,
                 max_doc_tokens: int = 180):
        # Force CPU for stability on macOS and avoid MPS issues
        self.device = torch.device("cpu")
        # Reduce thread contention / OpenMP issues
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        # Use generic AutoModel; colbertv2.0 exposes token embeddings via last_hidden_state
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).to(self.device)
        self.model.eval()
        self.max_query_tokens = max_query_tokens
        self.max_doc_tokens = max_doc_tokens

    @torch.no_grad()
    def _encode_tokens(self, texts: List[str], max_len: int) -> torch.Tensor:
        if not texts:
            return torch.empty(0)
        toks = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors='pt'
        ).to(self.device)
        out = self.model(**toks)
        # Expect last_hidden_state: [B, L, D]
        if hasattr(out, 'last_hidden_state'):
            reps = out.last_hidden_state
        elif isinstance(out, (list, tuple)):
            reps = out[0]
        else:
            # Fallback if model returns dict with 'pooler_output' only
            reps = out['last_hidden_state']
        # L2 normalize along embedding dim
        reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    @torch.no_grad()
    def _score_pairwise(self, q_rep: torch.Tensor, d_rep: torch.Tensor) -> float:
        """
        q_rep: [1, Lq, D], d_rep: [1, Ld, D]
        MaxSim over doc tokens, sum over query tokens
        """
        # remove CLS/SEP tokens if present by slicing interior tokens
        q_tokens = q_rep[:, 1:-1, :] if q_rep.size(1) > 2 else q_rep
        d_tokens = d_rep[:, 1:-1, :] if d_rep.size(1) > 2 else d_rep
        # [1, Lq, Ld]
        sims = torch.einsum('bid,bjd->bij', q_tokens, d_tokens)
        # max over doc tokens, then sum over query tokens
        max_per_q = sims.max(dim=-1).values  # [1, Lq]
        score = max_per_q.sum().item()
        return float(score)

    @torch.no_grad()
    def rerank(self,
               query: str,
               candidate_id_to_text: List[Tuple[str, str]],
               top_k: int = 10) -> List[str]:
        if not candidate_id_to_text:
            return []
        q_rep = self._encode_tokens([query], self.max_query_tokens)
        scores: List[Tuple[str, float]] = []
        for doc_id, text in candidate_id_to_text:
            d_rep = self._encode_tokens([text], self.max_doc_tokens)
            score = self._score_pairwise(q_rep, d_rep)
            scores.append((doc_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in scores[:top_k]]


