from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class MissionState(TypedDict, total=False):
    mission_id: str
    workflow_run_id: str
    actor_user_id: str
    mission: dict[str, Any]
    agent_registry: list[dict[str, Any]]
    test_mode: str
    workflow_profile: str

    complaint_csv_path: str
    operations_directory: str
    logistics_data_dir: str

    mission_plan: dict[str, Any]
    routing: list[dict[str, Any]]
    knowledge_result: dict[str, Any]
    complaint_result: dict[str, Any]
    root_cause_result: dict[str, Any]
    adversarial_result: dict[str, Any]
    logistics_plan_result: dict[str, Any]
    logistics_result: dict[str, Any]
    recovery_result: dict[str, Any]
    recovery_routing: list[dict[str, Any]]
    recovery_action: str
    failed_agent_code: str | None
    original_execution_agent_code: str | None
    failure_result: dict[str, Any]
    failed_agent_telemetry: dict[str, Any]
    failure_context: dict[str, Any]
    recovery_context: dict[str, Any]
    skipped_stages: list[str]
    recovery_required: bool
    stopped_reason: str | None
    verification_result: dict[str, Any]
    trust_graph: dict[str, Any]
    governance_result: dict[str, Any]
    evaluation_result: dict[str, Any]
    final_report: dict[str, Any]

    status: str
    current_stage: str
    progress_percent: float
    total_tokens: Annotated[int, operator.add]
    total_latency_ms: Annotated[int, operator.add]
    estimated_cost: Annotated[float, operator.add]

    approval_required: bool
    approval_id: str | None
    approval_status: str | None
    approval_comment: str | None

    retry_count: int
    revision_count: int
    stage_retry_counts: dict[str, int]
    failed_stage: str | None
    last_error: str | None
    errors: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], operator.add]
    timeline: Annotated[list[dict[str, Any]], operator.add]
