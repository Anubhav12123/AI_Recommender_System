# services/api/app/services/eval.py
from __future__ import annotations

import math
import random
import time
import importlib
import inspect
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from common import config  # project root must be on PYTHONPATH

# ------------------------------------------------------------------------------
# Column normalizer for ratings
# ------------------------------------------------------------------------------

def _normalize_ratings_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename common variants to expected names: user_id, item_id, rating, timestamp.
    Works with MovieLens-style (userId,movieId,rating,timestamp) and others.
    """
    # map lower->original
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
    ts_col = pick("timestamp", "Timestamp", "time", "ts")

    if not user_col or not item_col:
        missing = []
        if not user_col:
            missing.append("user_id (e.g., userId)")
        if not item_col:
            missing.append("item_id (e.g., itemId/movieId)")
        raise ValueError(f"ratings.csv missing required columns: {', '.join(missing)}")

    rename_map = {
        user_col: "user_id",
        item_col: "item_id",
    }
    if rating_col:
        rename_map[rating_col] = "rating"
    if ts_col:
        rename_map[ts_col] = "timestamp"

    return df.rename(columns=rename_map)

# ------------------------------------------------------------------------------
# CF adapter
# ------------------------------------------------------------------------------

try:
    from .cf_inference import CFInference as _CFInferenceClass  # optional
except Exception:
    _CFInferenceClass = None  # type: ignore

_cf_module = importlib.import_module("services.api.app.services.cf_inference")


class CFAdapter:
    """
    Unifies access to collaborative filtering recommenders.

    Works with either:
      1) A class `CFInference` exposing: recommend_for_user(user_id, k=?)
      2) A module-level function `recommend_for_user(user_id, k=?)`

    Supports common top-k parameter names: k / top_k / n.
    Normalizes outputs to: List[{'item_id': str, ...}].
    """

    def __init__(self):
        # pick implementation
        if _CFInferenceClass is not None:
            self._mode = "class"
            self._impl = _CFInferenceClass()
            self._callable = getattr(self._impl, "recommend_for_user", None)
        elif hasattr(_cf_module, "recommend_for_user"):
            self._mode = "func"
            self._impl = _cf_module
            self._callable = getattr(_cf_module, "recommend_for_user")
        else:
            raise ImportError(
                "Neither class `CFInference` nor function `recommend_for_user` "
                "found in services/api/app/services/cf_inference.py"
            )

        if not callable(self._callable):
            raise TypeError("`recommend_for_user` is not callable in cf_inference.py")

        # detect top-k parameter name
        sig = inspect.signature(self._callable)
        params = list(sig.parameters.keys())
        self._k_name: Optional[str] = None
        for cand in ("k", "top_k", "n"):
            if cand in params:
                self._k_name = cand
                break
        self._supports_k = self._k_name is not None

        # infer user_id expected type (optional)
        self._uid_cast = str
        try:
            uid_param = sig.parameters.get("user_id")
            if uid_param and uid_param.annotation is int:
                self._uid_cast = int  # type: ignore[assignment]
        except Exception:
            pass

    def recommend_for_user(self, user_id: str, k: int = 10) -> List[Dict[str, str]]:
        # cast user id if function expects int and the value is numeric
        uid = user_id
        if self._uid_cast is int and str(user_id).isdigit():
            uid = int(user_id)  # type: ignore[assignment]

        kwargs = {}
        if self._supports_k and self._k_name:
            kwargs[self._k_name] = k

        recs = self._callable(uid, **kwargs)

        # Normalize to list[{'item_id': str, ...}]
        if not recs:
            return []
        if isinstance(recs[0], dict) and "item_id" in recs[0]:
            # Ensure item_id is string
            return [{"item_id": str(r["item_id"]), **{k2: v2 for k2, v2 in r.items() if k2 != "item_id"}} for r in recs]

        # otherwise treat as list of ids
        return [{"item_id": str(x)} for x in recs]

# ------------------------------------------------------------------------------
# Holdout split + metrics
# ------------------------------------------------------------------------------

def _choose_holdout(df: pd.DataFrame) -> pd.DataFrame:
    """
    Leave-one-out per user:

    - If 'timestamp' exists -> take the latest interaction as the test item.
    - Else -> prefer a random positive (rating >= 4.0) if available; otherwise any.

    Returns: DataFrame with ['user_id', 'item_id'] (the test item per user).
    """
    df = df.copy()

    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    has_ts = "timestamp" in df.columns
    rows: List[Tuple[str, str]] = []

    for uid, g in df.groupby("user_id"):
        if len(g) < 2:
            continue

        if has_ts:
            g2 = g.sort_values("timestamp")
            test_row = g2.iloc[-1]
        else:
            if "rating" in g.columns and g["rating"].notna().any():
                pos = g[g["rating"] >= 4.0] if (g["rating"] >= 4.0).any() else g
                test_row = pos.sample(1, random_state=42).iloc[0]
            else:
                test_row = g.sample(1, random_state=42).iloc[0]

        rows.append((str(uid), str(test_row["item_id"])))

    return pd.DataFrame(rows, columns=["user_id", "item_id"])


def _dcg(rels: List[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def _ndcg_at_k(true_item: str, recs: List[str], k: int) -> float:
    hits = [1 if r == true_item else 0 for r in recs[:k]]
    ideal = sorted(hits, reverse=True)
    dcg = _dcg(hits)
    idcg = _dcg(ideal) or 1.0
    return dcg / idcg


def _ap_at_k(true_item: str, recs: List[str], k: int) -> float:
    score, hit_count = 0.0, 0
    for i, r in enumerate(recs[:k], start=1):
        if r == true_item:
            hit_count += 1
            score += hit_count / i
    return (score / hit_count) if hit_count else 0.0

# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------

def evaluate_cf(k: int = 10, users_limit: Optional[int] = None) -> Dict:
    """
    Offline evaluation of the CF recommender using leave-one-out per user.

    Reads interactions from `config.RATINGS_CSV`.
    Uses CF implementation provided by `services/.../cf_inference.py`
    (class `CFInference` or function `recommend_for_user`).

    Metrics:
      - precision@k, recall@k, MAP@k, NDCG@k, hit rate, item coverage,
        users evaluated, elapsed seconds
    """
    t0 = time.time()

    # Load and normalize ratings columns
    ratings_raw = pd.read_csv(config.RATINGS_CSV, dtype=str)
    ratings = _normalize_ratings_columns(ratings_raw)

    holdout = _choose_holdout(ratings)
    if len(holdout) == 0:
        raise ValueError("No eligible users for leave-one-out (need ≥ 2 interactions per user).")

    if users_limit:
        holdout = holdout.sample(min(users_limit, len(holdout)), random_state=42)

    cf = CFAdapter()

    precisions: List[float] = []
    recalls: List[float] = []
    maps: List[float] = []
    ndcgs: List[float] = []
    hits: List[int] = []
    coverage_items: set[str] = set()

    total = len(holdout)

    for _, row in holdout.iterrows():
        uid, true_item = row["user_id"], row["item_id"]

        # Get top-k recommendations for the user
        rec_items = [r["item_id"] for r in cf.recommend_for_user(uid, k=k)]
        coverage_items.update(rec_items)

        hit = 1 if true_item in rec_items[:k] else 0
        hits.append(hit)

        # In LOO (one relevant item), precision = hit/k, recall = hit/1
        precisions.append(hit / k)
        recalls.append(float(hit))
        maps.append(_ap_at_k(true_item, rec_items, k))
        ndcgs.append(_ndcg_at_k(true_item, rec_items, k))

    elapsed = time.time() - t0

    metrics = {
        "k": int(k),
        "users_evaluated": int(total),
        "precision_at_k": round(float(np.mean(precisions)) if precisions else 0.0, 4),
        "recall_at_k": round(float(np.mean(recalls)) if recalls else 0.0, 4),
        "map_at_k": round(float(np.mean(maps)) if maps else 0.0, 4),
        "ndcg_at_k": round(float(np.mean(ndcgs)) if ndcgs else 0.0, 4),
        "hit_rate": round(float(np.mean(hits)) if hits else 0.0, 4),
        "item_coverage": int(len(coverage_items)),
        "elapsed_sec": round(float(elapsed), 2),
    }
    return metrics
