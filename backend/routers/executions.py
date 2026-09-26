from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from sqlalchemy import select

from backend.dependencies import CurrentUserDependency, DatabaseDependency
from backend.models import Mission, WorkflowRun
from backend.schemas import (
    WorkflowRunResponse,
    WorkflowStartRequest,
    WorkflowStartResponse,
)
from backend.services.workflow_service import (
    DATASET_REQUIRED_MESSAGE,
    WorkflowService,
    detect_workflow_profile,
)


router = APIRouter(prefix="/api/v1", tags=["Workflow Execution"])
service = WorkflowService()


def _accessible_mission(
    database: DatabaseDependency,
    current_user: CurrentUserDependency,
    mission_id: UUID,
) -> Mission:
    mission = database.get(Mission, mission_id)
    if mission is None:
        raise HTTPException(status_code=404, detail="Mission not found.")
    if current_user.role not in {"administrator", "reviewer"}:
        if mission.created_by_id != current_user.id:
            raise HTTPException(status_code=403, detail="Mission access denied.")
    return mission


def _run_response(run: WorkflowRun) -> WorkflowRunResponse:
    snapshot = dict(run.state_snapshot or {})
    return WorkflowRunResponse(
        id=run.id,
        mission_id=run.mission_id,
        status=run.status,
        current_stage=run.current_stage,
        retry_count=run.retry_count,
        started_at=run.started_at,
        completed_at=run.completed_at,
        created_at=run.created_at,
        updated_at=run.updated_at,
        progress_percent=float(snapshot.get("progress_percent", 0.0) or 0.0),
        total_tokens=int(snapshot.get("total_tokens", 0) or 0),
        total_latency_ms=int(snapshot.get("total_latency_ms", 0) or 0),
        estimated_cost=float(snapshot.get("estimated_cost", 0.0) or 0.0),
        approval_required=bool(snapshot.get("approval_required", False)),
        approval_status=snapshot.get("approval_status"),
        timeline=list(snapshot.get("timeline", [])),
        state_snapshot=snapshot,
    )


@router.get("/missions/{mission_id}/data-readiness")
def get_mission_data_readiness(
    mission_id: UUID,
    database: DatabaseDependency,
    current_user: CurrentUserDependency,
) -> dict[str, Any]:
    mission = _accessible_mission(database, current_user, mission_id)
    return service.data_readiness(database, mission=mission)


@router.post(
    "/missions/{mission_id}/execute",
    response_model=WorkflowStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def execute_mission(
    mission_id: UUID,
    request: WorkflowStartRequest,
    background_tasks: BackgroundTasks,
    database: DatabaseDependency,
    current_user: CurrentUserDependency,
) -> WorkflowStartResponse:
    mission = _accessible_mission(database, current_user, mission_id)

    profile = detect_workflow_profile(mission)
    # Complaint analysis remains upload-only. Logistics uses the checked-in three-file
    # data package under sample_data/logistics and is validated by WorkflowService.
    if profile == "complaint" and request.complaint_document_id is None:
        raise HTTPException(status_code=422, detail=DATASET_REQUIRED_MESSAGE)

    active = database.scalar(
        select(WorkflowRun).where(
            WorkflowRun.mission_id == mission.id,
            WorkflowRun.status.in_(["queued", "running", "awaiting_approval"]),
        )
    )
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Mission already has an active workflow: {active.id}",
        )

    try:
        run = service.create_run(
            database,
            mission=mission,
            actor_user_id=current_user.id,
            complaint_document_id=request.complaint_document_id,
            operations_directory=request.operations_directory,
            test_mode=request.test_mode,
            use_default_data=False,
        )
        database.commit()
        database.refresh(run)
    except ValueError as exc:
        database.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    background_tasks.add_task(service.execute_run, run.id)
    return WorkflowStartResponse(
        workflow_run_id=run.id,
        mission_id=mission.id,
        status=run.status,
        message="CORTEX workflow queued successfully.",
    )


@router.get("/workflow-runs/{run_id}", response_model=WorkflowRunResponse)
def get_workflow_run(
    run_id: UUID,
    database: DatabaseDependency,
    current_user: CurrentUserDependency,
) -> WorkflowRunResponse:
    run = database.get(WorkflowRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Workflow run not found.")
    _accessible_mission(database, current_user, run.mission_id)
    return _run_response(run)


@router.get(
    "/missions/{mission_id}/workflow/latest",
    response_model=WorkflowRunResponse,
)
def get_latest_workflow(
    mission_id: UUID,
    database: DatabaseDependency,
    current_user: CurrentUserDependency,
) -> WorkflowRunResponse:
    _accessible_mission(database, current_user, mission_id)
    run = service.latest_run(database, mission_id)
    if run is None:
        raise HTTPException(status_code=404, detail="No workflow run exists.")
    return _run_response(run)


@router.get("/workflow-runs/{run_id}/report")
def get_final_report(
    run_id: UUID,
    database: DatabaseDependency,
    current_user: CurrentUserDependency,
) -> dict[str, Any]:
    run = database.get(WorkflowRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Workflow run not found.")
    _accessible_mission(database, current_user, run.mission_id)
    report = (run.state_snapshot or {}).get("final_report")
    if not report:
        raise HTTPException(status_code=404, detail="Final report is not ready.")
    return report


@router.get("/workflow-runs/{run_id}/trust-graph")
def get_trust_graph(
    run_id: UUID,
    database: DatabaseDependency,
    current_user: CurrentUserDependency,
) -> dict[str, Any]:
    run = database.get(WorkflowRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Workflow run not found.")
    _accessible_mission(database, current_user, run.mission_id)
    graph = (run.state_snapshot or {}).get("trust_graph")
    if not graph:
        raise HTTPException(status_code=404, detail="TrustGraph is not ready.")
    return graph


@router.get("/workflow-runs/{run_id}/artifacts")
def get_workflow_artifacts(
    run_id: UUID,
    database: DatabaseDependency,
    current_user: CurrentUserDependency,
) -> dict[str, Any]:
    run = database.get(WorkflowRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Workflow run not found.")
    _accessible_mission(database, current_user, run.mission_id)
    state = dict(run.state_snapshot or {})
    return {
        "mission_plan": state.get("mission_plan"),
        "routing": state.get("routing"),
        "knowledge": state.get("knowledge_result"),
        "complaint_intelligence": state.get("complaint_result"),
        "root_causes": state.get("root_cause_result"),
        "adversarial_test": state.get("adversarial_result"),
        "logistics_plan": state.get("logistics_plan_result"),
        "logistics_execution": state.get("logistics_result"),
        "recovery": state.get("recovery_result"),
        "recovery_routing": state.get("recovery_routing"),
        "verification": state.get("verification_result"),
        "trust_graph": state.get("trust_graph"),
        "governance": state.get("governance_result"),
        "evaluation": state.get("evaluation_result"),
        "final_report": state.get("final_report"),
        "timeline": state.get("timeline", []),
    }
