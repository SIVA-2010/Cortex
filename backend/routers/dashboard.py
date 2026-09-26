from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Query
from sqlalchemy import func, select

from backend.dependencies import CurrentUserDependency, DatabaseDependency
from backend.models import (
    Agent,
    AgentRun,
    AuditLog,
    EvaluationResult,
    Mission,
    WorkflowApproval,
    WorkflowRun,
)
from backend.schemas import AuditLogResponse, DashboardSummaryResponse


router = APIRouter(prefix="/api/v1/dashboard", tags=["Dashboard and Audit"])


@router.get("/summary", response_model=DashboardSummaryResponse)
def dashboard_summary(
    database: DatabaseDependency,
    current_user: CurrentUserDependency,
) -> DashboardSummaryResponse:
    mission_filter = []
    if current_user.role not in {"administrator", "reviewer"}:
        mission_filter.append(Mission.created_by_id == current_user.id)

    active_missions = int(
        database.scalar(
            select(func.count(Mission.id)).where(
                *mission_filter,
                Mission.status.in_(["queued", "running", "awaiting_approval"]),
            )
        )
        or 0
    )
    completed_missions = int(
        database.scalar(
            select(func.count(Mission.id)).where(
                *mission_filter, Mission.status == "completed"
            )
        )
        or 0
    )
    failed_missions = int(
        database.scalar(
            select(func.count(Mission.id)).where(
                *mission_filter, Mission.status == "failed"
            )
        )
        or 0
    )
    available_agents = int(
        database.scalar(
            select(func.count(Agent.id)).where(Agent.status == "active")
        )
        or 0
    )
    average_trust_score = float(
        database.scalar(select(func.avg(Agent.trust_score))) or 0.0
    )
    pending_approvals = int(
        database.scalar(
            select(func.count(WorkflowApproval.id)).where(
                WorkflowApproval.status == "pending"
            )
        )
        or 0
    )
    average_mission_score = float(
        database.scalar(select(func.avg(EvaluationResult.mission_success_score)))
        or 0.0
    )
    total_tokens = int(database.scalar(select(func.sum(AgentRun.total_tokens))) or 0)
    total_estimated_cost = float(
        database.scalar(select(func.sum(AgentRun.estimated_cost))) or 0.0
    )
    total_workflow_runs = int(
        database.scalar(select(func.count(WorkflowRun.id))) or 0
    )

    activity_query = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(8)
    if current_user.role not in {"administrator", "reviewer"}:
        accessible_ids = select(Mission.id).where(
            Mission.created_by_id == current_user.id
        )
        activity_query = activity_query.where(AuditLog.mission_id.in_(accessible_ids))
    activity = list(database.scalars(activity_query))

    return DashboardSummaryResponse(
        active_missions=active_missions,
        completed_missions=completed_missions,
        failed_missions=failed_missions,
        available_agents=available_agents,
        average_trust_score=round(average_trust_score, 2),
        pending_approvals=pending_approvals,
        average_mission_score=round(average_mission_score, 2),
        total_tokens=total_tokens,
        total_estimated_cost=round(total_estimated_cost, 6),
        total_workflow_runs=total_workflow_runs,
        recent_activity=[
            {
                "timestamp": item.created_at.isoformat(),
                "action": item.action,
                "mission_id": str(item.mission_id) if item.mission_id else None,
                "entity_type": item.entity_type,
                "details": item.details or {},
            }
            for item in activity
        ],
    )


@router.get("/audit-logs", response_model=list[AuditLogResponse])
def audit_logs(
    database: DatabaseDependency,
    current_user: CurrentUserDependency,
    mission_id: UUID | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[AuditLogResponse]:
    query = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)
    if mission_id is not None:
        query = query.where(AuditLog.mission_id == mission_id)
    if current_user.role not in {"administrator", "reviewer"}:
        accessible_ids = select(Mission.id).where(
            Mission.created_by_id == current_user.id
        )
        query = query.where(AuditLog.mission_id.in_(accessible_ids))
    records = list(database.scalars(query))
    return [
        AuditLogResponse(
            id=record.id,
            timestamp=record.created_at,
            actor_user_id=record.actor_user_id,
            mission_id=record.mission_id,
            task_id=record.task_id,
            action=record.action,
            entity_type=record.entity_type,
            entity_id=record.entity_id,
            details=record.details,
        )
        for record in records
    ]
