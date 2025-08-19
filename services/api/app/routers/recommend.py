# services/api/app/routers/recommend.py
from fastapi import APIRouter, HTTPException, Query
from typing import Any, Dict, List

# Your CF module exports CFInference + alias CFRecommender in your setup
from ..services.cf_inference import CFRecommender

router = APIRouter()
recommender = CFRecommender()

@router.get("/recommend/user")
def recommend_for_user(user_id: str = Query(...), k: int = Query(10, ge=1, le=100)) -> Dict[str, List[Dict[str, Any]]]:
    """
    Personalized recommendations for a user.
    Response: {"results": [{"item_id": "...", (optional) "score": ...}, ...]}
    """
    try:
        results = recommender.recommend_for_user(user_id=user_id, k=k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {e}")
