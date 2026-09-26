from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import text

from backend.dependencies import CurrentUserDependency, DatabaseDependency
from backend.services.llm import get_llm_service
from backend.services.vector_store import get_vector_store_service
from backend.services.workflow_service import get_workflow_runtime


router = APIRouter(prefix="/api/v1/system", tags=["System"])


@router.get("/components")
def component_health(
    database: DatabaseDependency,
    current_user: CurrentUserDependency,
) -> dict:
    database.execute(text("SELECT 1"))
    llm = get_llm_service().health_check()
    vector_store = get_vector_store_service().health_check()
    try:
        workflow = get_workflow_runtime().health()
    except Exception as exc:
        workflow = {"status": "failed", "error": str(exc)}

    statuses = ["ready", llm.get("status"), vector_store.get("status"), workflow.get("status")]
    return {
        "status": "ready" if all(value == "ready" for value in statuses) else "degraded",
        "database": {"status": "ready", "connection": "connected"},
        "llm": llm,
        "vector_store": vector_store,
        "workflow": workflow,
    }
