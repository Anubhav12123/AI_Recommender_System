# services/api/app/services/cf_inference.py
"""
Minimal CF inference that always works:

- Tries to load precomputed artifacts (optional).
- If artifacts are missing or incompatible, falls back to a fast popularity /
  lightweight item-based co-occurrence computed from ratings.csv.

Exports BOTH:
  - class CFInference with .recommend_for_user(user_id, k=10)
  - function recommend_for_user(user_id, k=10)
So your CFAdapter/eval will find at least one entry point.
"""

from __future__ import annotations

import os
from typing import Dict, List, Set, Tuple
import json

import numpy as np
import pandas as pd

from common import config


# ------------------------------
# Utilities
# ------------------------------

def _normalize_ratings_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep in sync with eval.py normalizer (simplified copy)."""
    lower_to_orig = {c.lower(): c for c in df.columns}

    def pick(*cands):
        for c in cands:
            key = c.lower()
            if key in lower_to_orig:
                return lower_to_orig[key]
        return None

    user_col = pick("user_id", "userid", "userId", "userID", "UserId", "UserID", "user")
    item_col = pick("item_id", "itemid", "itemId", "itemID", "movieId", "movie_id", "book_id", "productId", "product_id", "item")
    rating_col = pick("rating", "Rating", "score", "Score", "stars")

    if not user_col or not item_col:
        missing = []
        if not user_col:
            missing.append("user_id (e.g., userId)")
        if not item_col:
            missing.append("item_id (e.g., itemId/movieId)")
        raise ValueError(f"ratings.csv missing required columns: {', '.join(missing)}")

    rename_map = {user_col: "user_id", item_col: "item_id"}
    if rating_col:
        rename_map[rating_col] = "rating"

    df2 = df.rename(columns=rename_map)
    return df2


# ------------------------------
# Optional artifacts loader
# ------------------------------

def _try_load_artifacts() -> Tuple[np.ndarray | None, Dict[str, int] | None, Dict[int, str] | None]:
    """
    Attempt to load a pre-trained CF artifact set, if your trainer saved them.
    If anything is missing, return (None, None, None) and we'll fall back.
    Expected (example) layout:
      data/artifacts/models/cf/
        - user_ids.json  (list of user ids as strings)
        - item_ids.json  (list of item ids as strings)
        - scores.npy     (matrix [n_users, n_items]) OR some factorization
    Adjust this to your trainer’s actual outputs if different.
    """
    base = getattr(config, "CF_ARTIFACTS_DIR", "data/artifacts/models/cf")
    scores_path = os.path.join(base, "scores.npy")
    users_path = os.path.join(base, "user_ids.json")
    items_path = os.path.join(base, "item_ids.json")

    try:
        if not (os.path.exists(scores_path) and os.path.exists(users_path) and os.path.exists(items_path)):
            return None, None, None

        scores = np.load(scores_path)  # shape [n_users, n_items]
        user_ids = json.load(open(users_path))
        item_ids = json.load(open(items_path))

        # build maps
        user_to_idx = {str(u): i for i, u in enumerate(user_ids)}
        idx_to_item = {i: str(it) for i, it in enumerate(item_ids)}

        if scores.ndim != 2 or scores.shape != (len(user_to_idx), len(idx_to_item)):
            return None, None, None

        return scores, user_to_idx, idx_to_item

    except Exception:
        return None, None, None


# ------------------------------
# CF Inference (always available)
# ------------------------------

class CFInference:
    """
    Simple, robust recommender.

    Prefers pretrained artifacts (if present).
    Otherwise builds:
      - user -> set(items)
      - item popularity
      - light item-item co-occurrence for personalization

    recommend_for_user(user_id, k):
      1) If artifacts exist -> take top-k scores for that user, excluding seen items.
      2) Else:
         - If user is known -> score by co-occurrence with user's seen items (+ pop prior)
         - If user is new -> return top popular items
    """

    def __init__(self):
        # Try artifacts first
        self.scores, self.user_to_idx, self.idx_to_item = _try_load_artifacts()
        self.has_artifacts = self.scores is not None

        # Load ratings for fallback / masking
        ratings_raw = pd.read_csv(config.RATINGS_CSV, dtype=str)
        df = _normalize_ratings_columns(ratings_raw)

        # Keep only user_id, item_id
        df = df[["user_id", "item_id"]].copy()

        # Build user->items
        self.user_items: Dict[str, Set[str]] = {}
        for uid, g in df.groupby("user_id"):
            self.user_items[str(uid)] = set(map(str, g["item_id"].tolist()))

        # Popularity
        self.item_pop: Dict[str, int] = df["item_id"].astype(str).value_counts().to_dict()

        # Light co-occurrence (item -> {other_item: count})
        # This is intentionally simple to avoid heavy memory usage.
        self.co_counts: Dict[str, Dict[str, int]] = {}
        for items in self.user_items.values():
            items_list = list(items)
            for i in range(len(items_list)):
                a = items_list[i]
                acc = self.co_counts.setdefault(a, {})
                for j in range(len(items_list)):
                    if i == j:
                        continue
                    b = items_list[j]
                    acc[b] = acc.get(b, 0) + 1

        # If artifacts exist but we also want to exclude seen items, keep user_items handy.
        # Also build reverse map for items for fast mask if needed
        if self.has_artifacts:
            self.item_to_idx = {v: k for k, v in self.idx_to_item.items()}  # item_id -> col idx

    # -------- artifact path --------

    def _recs_from_artifacts(self, user_id: str, k: int) -> List[str]:
        """Return top-k item_ids for a known user using the scores matrix. Falls back if user unknown."""
        if user_id not in self.user_to_idx:
            # cold-start user -> fallback to popularity
            return self._popular_only(k)

        uidx = self.user_to_idx[user_id]
        row = self.scores[uidx].copy()  # [n_items]

        # mask seen items
        seen = self.user_items.get(user_id, set())
        if seen:
            for it in seen:
                j = self.item_to_idx.get(it)
                if j is not None:
                    row[j] = -np.inf  # exclude

        topk_idx = np.argpartition(row, -k)[-k:]  # unsorted top-k
        # sort descending
        topk_idx = topk_idx[np.argsort(row[topk_idx])[::-1]]
        return [self.idx_to_item[int(j)] for j in topk_idx if np.isfinite(row[int(j)])]

    # -------- fallback path --------

    def _popular_only(self, k: int) -> List[str]:
        """Top-k by global popularity."""
        return [it for it, _cnt in sorted(self.item_pop.items(), key=lambda x: (-x[1], x[0]))[:k]]

    def _cooccurrence_recs(self, user_id: str, k: int) -> List[str]:
        """Score items by co-occurrence with user's items + popularity prior."""
        seen = self.user_items.get(user_id, set())
        if not seen:
            return self._popular_only(k)

        scores: Dict[str, float] = {}

        # For every item the user has seen, add co-occurrence counts to candidates
        for a in seen:
            neigh = self.co_counts.get(a, {})
            for b, c in neigh.items():
                if b in seen:
                    continue
                scores[b] = scores.get(b, 0.0) + float(c)

        # Add small popularity prior to break ties
        for it, pop in self.item_pop.items():
            if it in seen:
                continue
            scores[it] = scores.get(it, 0.0) + 0.01 * float(pop)

        # Top-k
        return [it for it, _s in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]]

    # -------- public --------

    def recommend_for_user(self, user_id: str, k: int = 10) -> List[Dict[str, str]]:
        user_id = str(user_id)

        if self.has_artifacts:
            try:
                recs = self._recs_from_artifacts(user_id, k)
            except Exception:
                # artifacts present but incompatible; fall back gracefully
                recs = self._cooccurrence_recs(user_id, k)
        else:
            recs = self._cooccurrence_recs(user_id, k)

        # Normalize to list of dicts
        return [{"item_id": str(it)} for it in recs[:k]]


# ------------------------------
# Module-level function (also exported)
# ------------------------------

# Create a single shared instance so repeated calls are cheap.
CFRecommender = CFInference

def recommend_for_user(user_id: str, k: int = 10) -> List[Dict[str, str]]:
    """
    Functional entrypoint mirroring the class method.
    Your adapter will detect either the class or this function.
    """
    return CFRecommender.recommend_for_user(user_id, k=k)
