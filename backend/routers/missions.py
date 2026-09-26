from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.database import get_db
from backend.dependencies import get_current_user
from backend.models import Mission, Task, User
from backend.schemas import (
    MissionCreate,
    MissionDetailResponse,
    MissionResponse,
    MissionSubmissionResponse,
)
from backend.services.template_planner import create_template_plan
from backend.services.workflow_profile import classify_cortex_input


router = APIRouter(
    prefix="/api/v1/missions",
    tags=["Missions"],
)


SUPPORTED_MISSION_INTENTS = {"complaint", "logistics"}


def _selected_domain_profile(business_domain: str) -> str | None:
    """Map the explicit Create Mission dropdown choice to a supported workflow."""

    normalized = str(business_domain or "").strip().lower()
    if normalized == "customer complaint intelligence":
        return "complaint"
    if normalized == "logistics intelligence":
        return "logistics"
    return None


def _profile_label(profile: str) -> str:
    return (
        "Logistics Intelligence"
        if profile == "logistics"
        else "Customer Complaint Intelligence"
    )


def _domain_validation_message(*, selected_domain: str, detected_profile: str) -> str:
    selected_profile = _selected_domain_profile(selected_domain)
    detected_label = _profile_label(detected_profile)

    if selected_profile is None:
        return (
            f"The selected Business Domain '{selected_domain}' does not have an active "
            "CORTEX mission workflow in this build. Select Customer Complaint Intelligence "
            f"or Logistics Intelligence. Your objective was detected as {detected_label}."
        )

    selected_label = _profile_label(selected_profile)
    return (
        "Domain mismatch detected. "
        f"Your objective is a {detected_label} request, but the selected Business Domain "
        f"is {selected_label}. Select {detected_label} to continue."
    )


def _mission_detail_statement(mission_id: UUID):
    """Build the mission query required by the detail response schema."""
    return (
        select(Mission)
        .options(
            selectinload(Mission.tasks).selectinload(
                Task.assigned_agent
            )
        )
        .where(Mission.id == mission_id)
    )


def _get_mission_or_404(
    mission_id: UUID,
    database: Session,
) -> Mission:
    """Load one mission with tasks and assigned agents."""
    mission = database.scalar(
        _mission_detail_statement(mission_id)
    )

    if mission is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Mission not found.",
        )

    return mission


def _create_supported_mission(
    *,
    payload: MissionCreate,
    requested_profile: str,
    database: Session,
    current_user: User,
) -> Mission:
    """Persist a mission only after the Create Mission intent gate allows it."""

    resolved_domain = (
        "Logistics Intelligence"
        if requested_profile == "logistics"
        else "Customer Complaint Intelligence"
    )

    mission = Mission(
        title=payload.title,
        objective=payload.objective,
        business_domain=resolved_domain,
        priority=payload.priority,
        risk_tolerance=payload.risk_tolerance,
        max_budget=payload.max_budget,
        output_format=payload.output_format,
        human_approval_preference=(
            payload.human_approval_preference
        ),
        created_by_id=current_user.id,
        status="draft",
        plan_source="pending",
    )

    try:
        database.add(mission)
        database.flush()

        if payload.auto_plan:
            create_template_plan(
                database=database,
                mission=mission,
            )

        database.commit()

    except Exception:
        database.rollback()
        raise

    return _get_mission_or_404(
        mission.id,
        database,
    )


@router.post(
    "/submit",
    response_model=MissionSubmissionResponse,
)
def submit_create_mission(
    payload: MissionCreate,
    database: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> MissionSubmissionResponse:
    """Interpret the Business objective before deciding whether to create a mission.

    Conversation, CORTEX help and unsupported questions return a short CORTEX
    response without creating a Mission, Task or LangGraph workflow. Only complaint
    and logistics objectives proceed into the existing mission architecture.
    """

    decision = classify_cortex_input(payload.objective)
    intent = str(decision["intent"])

    if intent not in SUPPORTED_MISSION_INTENTS:
        return MissionSubmissionResponse(
            intent=intent,
            mission_created=False,
            message=str(decision["message"]),
            workflow_profile=None,
            mission=None,
            intent_provider=str(decision.get("provider") or "local"),
            intent_tokens=int(decision.get("tokens") or 0),
            intent_latency_ms=int(decision.get("latency_ms") or 0),
        )

    selected_profile = _selected_domain_profile(payload.business_domain)
    if selected_profile != intent:
        return MissionSubmissionResponse(
            intent="domain_mismatch",
            mission_created=False,
            message=_domain_validation_message(
                selected_domain=payload.business_domain,
                detected_profile=intent,
            ),
            workflow_profile=None,
            mission=None,
            intent_provider=str(decision.get("provider") or "local"),
            intent_tokens=int(decision.get("tokens") or 0),
            intent_latency_ms=int(decision.get("latency_ms") or 0),
        )

    mission = _create_supported_mission(
        payload=payload,
        requested_profile=selected_profile,
        database=database,
        current_user=current_user,
    )

    selected_label = _profile_label(selected_profile)
    message = (
        f"Mission created. CORTEX validated the objective against the selected "
        f"{selected_label} domain."
    )

    return MissionSubmissionResponse(
        intent=intent,
        mission_created=True,
        message=message,
        workflow_profile=selected_profile,
        mission=mission,
        intent_provider=str(decision.get("provider") or "local"),
        intent_tokens=int(decision.get("tokens") or 0),
        intent_latency_ms=int(decision.get("latency_ms") or 0),
    )


@router.post(
    "",
    response_model=MissionDetailResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_mission(
    payload: MissionCreate,
    database: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Mission:
    """Create a supported business mission.

    This legacy endpoint remains available for API clients, but it now applies the
    same intent guard so unrelated text cannot silently become a complaint mission.
    """

    decision = classify_cortex_input(payload.objective)
    intent = str(decision["intent"])
    if intent not in SUPPORTED_MISSION_INTENTS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(decision["message"]),
        )

    selected_profile = _selected_domain_profile(payload.business_domain)
    if selected_profile != intent:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=_domain_validation_message(
                selected_domain=payload.business_domain,
                detected_profile=intent,
            ),
        )

    return _create_supported_mission(
        payload=payload,
        requested_profile=selected_profile,
        database=database,
        current_user=current_user,
    )


@router.get(
    "",
    response_model=list[MissionResponse],
)
def list_missions(
    database: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[Mission]:
    """Return missions with task collections loaded for task_count."""
    statement = (
        select(Mission)
        .options(
            selectinload(Mission.tasks).selectinload(
                Task.assigned_agent
            )
        )
        .order_by(Mission.created_at.desc())
    )

    return list(
        database.scalars(statement).all()
    )


@router.get(
    "/{mission_id}",
    response_model=MissionDetailResponse,
)
def get_mission(
    mission_id: UUID,
    database: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Mission:
    """Return one mission with tasks and assigned-agent details."""
    return _get_mission_or_404(
        mission_id,
        database,
    )


@router.post(
    "/{mission_id}/plan",
    response_model=MissionDetailResponse,
)
def plan_mission(
    mission_id: UUID,
    database: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Mission:
    """Regenerate the initial task plan for an existing mission."""
    mission = database.get(Mission, mission_id)

    if mission is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Mission not found.",
        )

    try:
        create_template_plan(
            database=database,
            mission=mission,
            replace_existing=True,
        )
        database.commit()

    except Exception:
        database.rollback()
        raise

    return _get_mission_or_404(
        mission_id,
        database,
    )
