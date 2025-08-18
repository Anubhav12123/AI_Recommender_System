# services/api/app/routers/search.py
from fastapi import APIRouter, HTTPException, Query
from typing import Any, Dict, List, Optional

from ..services.hybrid_ranker import HybridRanker

router = APIRouter()
ranker = HybridRanker()  # lazy loads artifacts internally

@router.get("/search")
def search(q: str = Query(..., min_length=1), k: int = Query(20, ge=1, le=100)) -> Dict[str, List[Dict[str, Any]]]:
    """
    Hybrid search: lexical + vector. Returns top-k items.
    Response shape matches the Streamlit client: {"results": [...]}
    """
    try:
        results = ranker.search(q, k=k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

@router.get("/search/similar")
def similar(item_id: str = Query(...), k: int = Query(10, ge=1, le=100)) -> Dict[str, List[Dict[str, Any]]]:
    """
    Item-to-item similarity using the embedding store.
    """
    try:
        results = ranker.similar(item_id=item_id, k=k)
        return {"results": results}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Item not found: {item_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similar failed: {e}")
