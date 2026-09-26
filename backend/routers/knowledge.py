from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.agents.knowledge_agent import KnowledgeRetrievalAgent
from backend.dependencies import CurrentUserDependency
from backend.schemas import KnowledgeSearchResponse
from backend.services.vector_store import get_vector_store_service


router = APIRouter(prefix="/api/v1/knowledge", tags=["Enterprise Knowledge"])


@router.get("/status")
def knowledge_status(current_user: CurrentUserDependency) -> dict:
    return get_vector_store_service().health_check()


@router.get("/search", response_model=KnowledgeSearchResponse)
def search_knowledge(
    current_user: CurrentUserDependency,
    q: str = Query(min_length=3, max_length=1000),
    limit: int = Query(default=5, ge=1, le=10),
) -> KnowledgeSearchResponse:
    result = KnowledgeRetrievalAgent().run(query=q, n_results=limit)
    if result.status != "completed":
        raise HTTPException(status_code=500, detail="; ".join(result.errors))
    return KnowledgeSearchResponse(
        query=q,
        answer=str(result.output.get("answer", "")),
        confidence=result.confidence,
        evidence=list(result.output.get("evidence", [])),
        tokens=result.total_tokens,
        latency_ms=result.latency_ms,
    )
