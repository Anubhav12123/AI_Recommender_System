from pydantic import BaseModel
from typing import List, Optional

class SearchHit(BaseModel):
    item_id: str
    title: str
    score: float
    reasons: List[str] = []

class SearchResponse(BaseModel):
    query: str
    hits: List[SearchHit]

class SimilarItemsResponse(BaseModel):
    item_id: str
    hits: List[SearchHit]

class UserRecsResponse(BaseModel):
    user_id: str
    hits: List[SearchHit]

class FeedbackEvent(BaseModel):
    user_id: str
    item_id: str
    action: str  # "click" | "view" | "like" | "dismiss"
    context: Optional[dict] = None
