# services/api/app/services/embedding_store.py
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors  # <-- ensure available

from common import config

class EmbeddingStore:
    """
    Loads precomputed item embeddings and provides item-to-item similarity.
    Expects:
      - data/artifacts/models/embeddings/item_vectors.npy
      - data/artifacts/models/embeddings/id_mapping.json  (index -> item_id)
    """

    def __init__(self):
        self.emb = None
        self.id_map: Dict[int, str] = {}
        self.nn: NearestNeighbors | None = None
        self._load()

    def _load(self):
        base = getattr(config, "EMB_DIR", "data/artifacts/models/embeddings")
        vec_path = os.path.join(base, "item_vectors.npy")
        map_path = os.path.join(base, "id_mapping.json")

        if not (os.path.exists(vec_path) and os.path.exists(map_path)):
            # embeddings not built; we’ll just disable similarity
            self.emb = None
            self.nn = None
            self.id_map = {}
            return

        self.emb = np.load(vec_path)
        raw_map = json.load(open(map_path))
        # ensure keys are ints
        self.id_map = {int(k): str(v) for k, v in raw_map.items()}

        self.nn = NearestNeighbors(metric="cosine")
        self.nn.fit(self.emb)

    def similar(self, item_id: str, k: int = 10) -> List[str]:
        if self.emb is None or self.nn is None:
            return []  # no embeddings available

        # find row index for item_id
        # id_map is idx -> item_id, build reverse on the fly
        rev = {v: k for k, v in self.id_map.items()}
        idx = rev.get(str(item_id))
        if idx is None:
            return []

        vec = self.emb[idx].reshape(1, -1)
        distances, indices = self.nn.kneighbors(vec, n_neighbors=min(k + 1, self.emb.shape[0]))
        # skip the item itself
        out: List[str] = []
        for i in indices[0]:
            if i == idx:
                continue
            out.append(self.id_map[int(i)])
            if len(out) >= k:
                break
        return out
