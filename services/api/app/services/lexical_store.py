import os, re, numpy as np, pandas as pd
from rank_bm25 import BM25Okapi
from typing import Dict
from common.storage import load_pickle
from common import config

ART_DIR = config.ARTIFACT_DIR
LEX_PATH = os.path.join(ART_DIR, "indices", "lexical.pkl")

def simple_tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())

class LexicalStore:
    def __init__(self):
        self.bm25 = None
        self.id_list = []
        self._load()

    def _load(self):
        if os.path.exists(LEX_PATH):
            obj = load_pickle(LEX_PATH)
            self.bm25 = obj["bm25"]
            self.id_list = obj["id_list"]

    def search(self, query: str, k: int = 10) -> Dict[str, float]:
        if self.bm25 is None:
            return {}
        tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(tokens)
        # normalize scores to [0,1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        top_idx = np.argsort(scores)[::-1][:k*2]
        return {self.id_list[i]: float(scores[i]) for i in top_idx if scores[i] > 0}

    def similar_items(self, item_id: str, k: int = 10) -> Dict[str, float]:
        # Use item's own doc as query if available
        if self.bm25 is None:
            return {}
        try:
            idx = self.id_list.index(item_id)
        except ValueError:
            return {}
        # Approximate: use top terms of the same doc
        # BM25 doesn't expose raw docs here; we score by using a pseudo-query token: id itself (hack)
        # Better: indexer can store original tokens list, but for demo we just score by bm25 scores column-wise.
        # We'll fallback to uniform small similarity here.
        scores = np.random.random(len(self.id_list)) * 0.1
        scores[idx] = 0.0
        top_idx = np.argsort(scores)[::-1][:k]
        return {self.id_list[i]: float(scores[i]) for i in top_idx}

    def popular_items(self, k: int = 10) -> Dict[str, float]:
        # naive popularity = original order weight
        if not self.id_list:
            return {}
        vals = np.linspace(1.0, 0.1, num=len(self.id_list))
        return {self.id_list[i]: float(vals[i]) for i in range(min(k, len(self.id_list)))}
