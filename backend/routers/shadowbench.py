from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from backend.dependencies import DatabaseDependency, ReviewerDependency
from backend.models import ShadowRun, WorkflowRun
from backend.schemas import ShadowRunResponse
from backend.services.shadowbench_service import ShadowBenchService


router = APIRouter(prefix="/api/v1/shadowbench", tags=["ShadowBench"])
service = ShadowBenchService()


@router.post(
    "/workflow-runs/{workflow_run_id}/replay",
    response_model=ShadowRunResponse,
    status_code=status.HTTP_201_CREATED,
)
def run_shadow_replay(
    workflow_run_id: UUID,
    database: DatabaseDependency,
    current_user: ReviewerDependency,
) -> ShadowRun:
    if database.get(WorkflowRun, workflow_run_id) is None:
        raise HTTPException(status_code=404, detail="Workflow run not found.")
    try:
        record = service.run_report_replay(
            database, workflow_run_id=workflow_run_id
        )
        database.commit()
        database.refresh(record)
        return record
    except ValueError as exc:
        database.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/workflow-runs/{workflow_run_id}",
    response_model=list[ShadowRunResponse],
)
def list_shadow_runs(
    workflow_run_id: UUID,
    database: DatabaseDependency,
    current_user: ReviewerDependency,
) -> list[ShadowRun]:
    return list(
        database.scalars(
            select(ShadowRun)
            .where(ShadowRun.workflow_run_id == workflow_run_id)
            .order_by(ShadowRun.created_at.desc())
        )
    )
