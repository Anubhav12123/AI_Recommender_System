# services/api/app/routers/metrics.py
from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from ..services.eval import evaluate_cf

router = APIRouter(prefix="/metrics", tags=["metrics"])

@router.get("/cf")
def cf_metrics(
    k: int = Query(10, ge=1, le=100),
    sample_users: Optional[int] = Query(None, ge=10)
):
    try:
        return evaluate_cf(k=k, users_limit=sample_users)
    except Exception as e:
        # Return readable error to UI instead of 500
        raise HTTPException(status_code=400, detail=f"Evaluation failed: {e}")
