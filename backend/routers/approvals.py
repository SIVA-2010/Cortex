from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from backend.dependencies import DatabaseDependency, ReviewerDependency
from backend.models import Mission, WorkflowApproval
from backend.schemas import ApprovalDecisionRequest, ApprovalResponse
from backend.services.workflow_persistence import WorkflowPersistenceService
from backend.services.workflow_service import WorkflowService


router = APIRouter(prefix="/api/v1/approvals", tags=["Human Approval"])
persistence = WorkflowPersistenceService()
workflow_service = WorkflowService()


def _response(record: WorkflowApproval, title: str | None = None) -> ApprovalResponse:
    return ApprovalResponse(
        id=record.id,
        workflow_run_id=record.workflow_run_id,
        mission_id=record.mission_id,
        mission_title=title,
        reason=record.reason,
        action_payload=record.action_payload,
        risk_level=record.risk_level,
        status=record.status,
        requested_by_agent=record.requested_by_agent,
        reviewer_id=record.reviewer_id,
        review_comment=record.review_comment,
        reviewed_at=record.reviewed_at,
        created_at=record.created_at,
    )


@router.get("/pending", response_model=list[ApprovalResponse])
def pending_approvals(
    database: DatabaseDependency,
    current_user: ReviewerDependency,
) -> list[ApprovalResponse]:
    records = list(
        database.scalars(
            select(WorkflowApproval)
            .where(WorkflowApproval.status == "pending")
            .order_by(WorkflowApproval.created_at.desc())
        )
    )
    titles = {
        mission.id: mission.title
        for mission in database.scalars(
            select(Mission).where(
                Mission.id.in_([record.mission_id for record in records])
            )
        )
    } if records else {}
    return [_response(record, titles.get(record.mission_id)) for record in records]


@router.post("/{approval_id}/decision", response_model=ApprovalResponse)
def decide_approval(
    approval_id: UUID,
    request: ApprovalDecisionRequest,
    background_tasks: BackgroundTasks,
    database: DatabaseDependency,
    current_user: ReviewerDependency,
) -> ApprovalResponse:
    record = database.get(WorkflowApproval, approval_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Approval request not found.")
    if record.status != "pending":
        raise HTTPException(status_code=409, detail="Approval is already resolved.")

    record = persistence.resolve_approval(
        database,
        approval_id=approval_id,
        reviewer_id=current_user.id,
        action=request.action,
        comment=request.comment,
    )
    mission = database.get(Mission, record.mission_id)
    if mission is not None:
        mission.status = "running"
    database.commit()
    database.refresh(record)

    background_tasks.add_task(
        workflow_service.resume_run,
        workflow_run_id=record.workflow_run_id,
        action=request.action,
        comment=request.comment,
    )
    return _response(record, mission.title if mission else None)
