from fastapi import APIRouter, status
from common.schemas import FeedbackEvent
import logging
router = APIRouter()
log = logging.getLogger(__name__)

@router.post("/", status_code=status.HTTP_202_ACCEPTED)
def post_feedback(event: FeedbackEvent):
    # In real system: send to Kafka; here we just log
    log.info("FEEDBACK %s", event.model_dump())
    return {"accepted": True}
