from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserResponse(BaseModel):
    id: UUID
    email: EmailStr
    full_name: str
    role: str
    is_active: bool

    model_config = ConfigDict(from_attributes=True)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class MessageResponse(BaseModel):
    message: str


class MissionCreate(BaseModel):
    title: str = Field(min_length=3, max_length=200)
    objective: str = Field(min_length=1, max_length=5000)
    business_domain: str = Field(
        default="Customer Complaint Intelligence", min_length=3, max_length=100
    )
    priority: Literal["low", "medium", "high", "critical"] = "high"
    risk_tolerance: Literal["conservative", "balanced", "experimental"] = (
        "balanced"
    )
    max_budget: float | None = Field(default=25.0, ge=0, le=100000)
    output_format: Literal["executive_report", "detailed_report", "json"] = (
        "executive_report"
    )
    human_approval_preference: bool = True
    auto_plan: bool = True


class TaskResponse(BaseModel):
    id: UUID
    mission_id: UUID
    title: str
    description: str
    task_type: str
    sequence_order: int
    execution_mode: str
    priority: str
    risk_level: str
    status: str
    expected_output: str
    assigned_agent_id: UUID | None
    assigned_agent_name: str | None
    assigned_agent_code: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MissionResponse(BaseModel):
    id: UUID
    title: str
    objective: str
    business_domain: str
    priority: str
    risk_tolerance: str
    max_budget: float | None
    output_format: str
    human_approval_preference: bool
    status: str
    plan_source: str
    created_by_id: UUID
    task_count: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MissionDetailResponse(MissionResponse):
    tasks: list[TaskResponse]


class MissionSubmissionResponse(BaseModel):
    intent: Literal[
        "conversation",
        "cortex_help",
        "complaint",
        "logistics",
        "unsupported",
        "domain_mismatch",
    ]
    mission_created: bool
    message: str
    workflow_profile: Literal["complaint", "logistics"] | None = None
    mission: MissionDetailResponse | None = None
    intent_provider: str = "local"
    intent_tokens: int = 0
    intent_latency_ms: int = 0


class AgentResponse(BaseModel):
    id: UUID
    code: str
    name: str
    description: str
    capabilities: list[str]
    supported_task_types: list[str]
    model_provider: str
    model_deployment: str
    status: str
    trust_score: float
    reliability_score: float
    average_latency_ms: int
    hallucination_rate: float
    completed_runs: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class WorkflowStartRequest(BaseModel):
    complaint_document_id: UUID | None = None
    complaint_csv_path: str | None = None
    operations_directory: str | None = None
    # Per-run control. Complaint workflows always enable the fixed adversarial
    # assurance stage in WorkflowService; the legacy normal/adversarial values are
    # retained for API compatibility. Logistics keeps explicit failure-test modes.
    test_mode: Literal[
        "normal",
        "adversarial",
        "logistics_reroute_failure",
        "logistics_stop_failure",
    ] = "normal"
    # Normal API execution never falls back to bundled sample data. A dataset
    # must be uploaded and linked to the mission before execution begins.
    use_default_data: bool = False


class WorkflowStartResponse(BaseModel):
    workflow_run_id: UUID
    mission_id: UUID
    status: str
    message: str


class WorkflowRunResponse(BaseModel):
    id: UUID
    mission_id: UUID
    status: str
    current_stage: str | None
    retry_count: int
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime
    progress_percent: float = 0.0
    total_tokens: int = 0
    total_latency_ms: int = 0
    estimated_cost: float = 0.0
    approval_required: bool = False
    approval_status: str | None = None
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    state_snapshot: dict[str, Any] = Field(default_factory=dict)


class ApprovalDecisionRequest(BaseModel):
    action: Literal["approve", "reject", "request_revision"]
    comment: str = Field(default="", max_length=2000)


class ApprovalResponse(BaseModel):
    id: UUID
    workflow_run_id: UUID
    mission_id: UUID
    mission_title: str | None = None
    reason: str
    action_payload: dict[str, Any]
    risk_level: str
    status: str
    requested_by_agent: str
    reviewer_id: UUID | None
    review_comment: str | None
    reviewed_at: datetime | None
    created_at: datetime


class UploadResponse(BaseModel):
    document_id: UUID
    mission_id: UUID | None
    file_name: str
    document_type: str
    size_bytes: int
    status: str
    message: str


class KnowledgeSearchResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    evidence: list[dict[str, Any]]
    tokens: int
    latency_ms: int


class DashboardSummaryResponse(BaseModel):
    active_missions: int
    completed_missions: int
    failed_missions: int
    available_agents: int
    average_trust_score: float
    pending_approvals: int
    average_mission_score: float
    total_tokens: int
    total_estimated_cost: float
    total_workflow_runs: int
    recent_activity: list[dict[str, Any]]


class AuditLogResponse(BaseModel):
    id: UUID
    timestamp: datetime
    actor_user_id: UUID | None
    mission_id: UUID | None
    task_id: UUID | None
    action: str
    entity_type: str | None
    entity_id: str | None
    details: dict[str, Any] | None


class ComponentHealthResponse(BaseModel):
    status: str
    database: dict[str, Any]
    llm: dict[str, Any]
    vector_store: dict[str, Any]
    workflow: dict[str, Any]


class ShadowRunResponse(BaseModel):
    id: UUID
    workflow_run_id: UUID
    production_agent_code: str
    shadow_agent_code: str
    production_deployment: str
    shadow_deployment: str
    status: str
    production_metrics: dict[str, Any]
    shadow_metrics: dict[str, Any]
    recommendation: str
    shadow_output: dict[str, Any] | None
    started_at: datetime
    completed_at: datetime | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
