import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from app.schemas import ChatRequest, ChatResponse, HistoryResponse
from app.services import history
from app.services.rag import LLMRateLimited, LLMUnavailable, PolicyNotIngested, answer_question

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    recent: list[dict] = []
    if req.session_id:
        try:
            recent = history.get_history(req.session_id, limit=3)
        except history.HistoryUnavailable:
            pass

    try:
        result = answer_question(question, recent)
    except PolicyNotIngested as exc:
        raise HTTPException(status_code=503, detail=f"{exc} Run the ingestion first.") from exc
    except LLMRateLimited as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except LLMUnavailable as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    now = datetime.now(timezone.utc)
    try:
        history.save_turn(
            question=question,
            answer=result.answer,
            policy_references=result.policy_references,
            recommended_action=result.recommended_action,
            found_in_policy=result.found_in_policy,
            session_id=req.session_id,
            timestamp=now,
            language=result.language,
        )
    except history.HistoryUnavailable as exc:
        # The answer is still useful to the employee even if history can't be saved.
        logger.warning("Could not save chat history: %s", exc)

    return ChatResponse(
        answer=result.answer,
        policy_references=result.policy_references,
        recommended_action=result.recommended_action,
        found_in_policy=result.found_in_policy,
        session_id=req.session_id,
        timestamp=now,
    )


@router.get("/history", response_model=HistoryResponse)
def get_history(
    session_id: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=50, ge=1, le=200),
) -> HistoryResponse:
    try:
        return HistoryResponse(items=history.get_history(session_id, limit))
    except history.HistoryUnavailable as exc:
        raise HTTPException(status_code=503, detail="Chat history database is unavailable.") from exc


@router.delete("/history")
def clear_history(session_id: str = Query(..., min_length=1, max_length=64)) -> dict:
    try:
        return {"deleted": history.delete_session(session_id)}
    except history.HistoryUnavailable as exc:
        raise HTTPException(status_code=503, detail="Chat history database is unavailable.") from exc
