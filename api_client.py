# services/ui/utils/api_client.py
from __future__ import annotations

import os
import requests
from typing import Any, Dict, List, Optional

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

def _get(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 120):
    url = f"{API_BASE}{path}"
    r = requests.get(url, params=params or {}, timeout=timeout)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except Exception:
            detail = r.text
        raise RuntimeError(f"API error {r.status_code}: {detail}")
    return r.json()

# ---------- Metrics (already working) ----------
def get_cf_metrics(k: int = 10, sample_users: int | None = None):
    params: Dict[str, Any] = {"k": k}
    if sample_users:
        params["sample_users"] = sample_users
    return _get("/metrics/cf", params=params, timeout=180)

# ---------- Search ----------
def search(query: str, k: int = 20) -> List[Dict[str, Any]]:
    """
    Calls backend search endpoint.
    Expected FastAPI route: GET /search?q=...&k=...
    Returns a list of items.
    """
    if not query:
        return []
    data = _get("/search", params={"q": query, "k": k}, timeout=60)
    # accept either {"results":[...]} or list
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    if isinstance(data, list):
        return data
    return []

def similar(item_id: str, k: int = 10) -> List[Dict[str, Any]]:
    """
    Calls backend item-similarity endpoint.
    Expected route: GET /search/similar?item_id=...&k=...
    """
    if not item_id:
        return []
    data = _get("/search/similar", params={"item_id": item_id, "k": k}, timeout=60)
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    if isinstance(data, list):
        return data
    return []

# ---------- Recommendations ----------
def user_recs(user_id: str, k: int = 10) -> List[Dict[str, Any]]:
    """
    Calls backend user recommendation endpoint.
    Expected route: GET /recommend/user?user_id=...&k=...
    """
    if not user_id:
        return []
    data = _get("/recommend/user", params={"user_id": user_id, "k": k}, timeout=60)
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    if isinstance(data, list):
        return data
    return []
