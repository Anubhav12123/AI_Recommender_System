# services/api/app/services/hybrid_ranker.py
from __future__ import annotations

from typing import Any, Dict, List
import re

from rank_bm25 import BM25Okapi

from ..repositories.items_repo import ItemsRepo
from .embedding_store import EmbeddingStore

TOKENIZER = re.compile(r"[A-Za-z0-9]+")

def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKENIZER.findall(text or "")]

class HybridRanker:
    """
    Simple, robust hybrid:
      - Lexical search = BM25 over items.text (built at startup)
      - Similar(item_id) = embedding_store nearest neighbors (if available)
    """

    def __init__(self):
        # Items
        self.items = ItemsRepo()

        # --- Lexical BM25 ---
        corpus = self.items.df["text"].tolist()
        tokenized_corpus = [ _tokenize(t) for t in corpus ]
        # If corpus is empty (shouldn’t happen), keep bm25 None to avoid crashes
        self.bm25 = BM25Okapi(tokenized_corpus) if len(tokenized_corpus) > 0 else None

        # --- Embeddings for item-to-item ---
        self.embed = EmbeddingStore()

        # Keep a quick row -> item_id map
        self.idx_to_item = self.items.df["item_id"].astype(str).tolist()

    # -------- public APIs used by routers --------

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        """
        Returns: list of dicts with keys item_id, title, description, score
        """
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        if self.bm25 is None:
            # Fallback: naive contains match on text
            mask = self.items.df["text"].str.contains(" ".join(q_tokens), na=False)
            sub = self.items.df.loc[mask].head(k)
            out = []
            for _, row in sub.iterrows():
                out.append({
                    "item_id": str(row["item_id"]),
                    "title": str(row.get("title", "")),
                    "description": str(row.get("description", "")),
                    "score": 0.0,
                })
            return out

        scores = self.bm25.get_scores(q_tokens)  # numpy array
        # get top-k indices
        if len(scores) == 0:
            return []

        top_idx = scores.argsort()[::-1][:k]
        results: List[Dict[str, Any]] = []
        for i in top_idx:
            row = self.items.df.iloc[int(i)]
            results.append({
                "item_id": str(row["item_id"]),
                "title": str(row.get("title", "")),
                "description": str(row.get("description", "")),
                "score": float(scores[int(i)]),
            })
        return results

    def similar(self, item_id: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Returns list of items similar to given item_id using embedding neighbors.
        Falls back to empty list if embeddings aren’t available.
        """
        neigh_ids = self.embed.similar(item_id=item_id, k=k)
        out: List[Dict[str, Any]] = []
        for nid in neigh_ids:
            item = self.items.get_item(nid)
            if item:
                out.append(item | {"score": None})
        return out
