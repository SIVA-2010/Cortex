from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.models import (
    Agent,
    AgentRun,
    ClaimRecord,
    EvaluationResult,
    EvidenceRecord,
    GovernanceDecision,
    Mission,
    Task,
    TaskDependency,
    WorkflowApproval,
    WorkflowRun,
)
from backend.services.audit_service import AuditService


STAGE_TO_TASK_TYPE = {
    "plan_mission": "mission_planning",
    "route_agents": "agent_routing",
    "retrieve_knowledge": "knowledge_retrieval",
    "analyse_complaints": "complaint_analysis",
    "analyse_root_causes": "root_cause_analysis",
    "inject_adversarial_claim": "adversarial_testing",
    "analyse_logistics": "logistics_routing",
    "execute_logistics": "logistics_execution",
    "execute_logistics_fallback": "logistics_execution",
    "recover_logistics": "workflow_recovery",
    "verify_claims": "claim_verification",
    "apply_governance": "governance_review",
    "evaluate_quality": "quality_evaluation",
    "generate_report": "report_generation",
}


class WorkflowPersistenceService:
    @staticmethod
    def _uuid(value: str | UUID) -> UUID:
        return value if isinstance(value, UUID) else UUID(str(value))

    @staticmethod
    def _clamp_score(value: float) -> float:
        return round(min(max(float(value), 0.0), 100.0), 2)

    @staticmethod
    def _execution_learning_rate(completed_runs: int) -> float:
        # New agents learn faster; established agents move more gradually.
        return max(0.04, min(0.10, 1.0 / max(completed_runs + 4, 1)))

    @staticmethod
    def _verification_learning_rate(completed_runs: int) -> float:
        # Evidence feedback is stronger than ordinary execution telemetry.
        return max(0.08, min(0.20, 2.0 / max(completed_runs + 6, 1)))

    @classmethod
    def _apply_execution_performance(
        cls,
        agent: Agent,
        agent_run: AgentRun,
    ) -> dict[str, float]:
        previous_runs = int(agent.completed_runs or 0)
        old_reliability = float(agent.reliability_score or 80.0)
        old_trust = float(agent.trust_score or 80.0)
        old_hallucination = float(agent.hallucination_rate or 0.0)
        old_latency = int(agent.average_latency_ms or 0)

        confidence_pct = cls._clamp_score(float(agent_run.confidence or 0.0) * 100.0)
        status_score = {
            "completed": 100.0,
            "awaiting_approval": 75.0,
            "failed": 20.0,
        }.get(str(agent_run.status).lower(), 40.0)

        warning_penalty = min(len(agent_run.warnings or []) * 1.5, 7.5)
        error_penalty = min(len(agent_run.errors or []) * 8.0, 24.0)
        execution_outcome = cls._clamp_score(
            status_score * 0.65
            + confidence_pct * 0.35
            - warning_penalty
            - error_penalty
        )

        alpha = cls._execution_learning_rate(previous_runs)
        new_reliability = cls._clamp_score(
            old_reliability + alpha * (execution_outcome - old_reliability)
        )

        safety_score = cls._clamp_score(100.0 - old_hallucination)
        trust_target = cls._clamp_score(
            new_reliability * 0.55
            + confidence_pct * 0.30
            + safety_score * 0.15
        )
        if str(agent_run.status).lower() == "failed":
            trust_target = cls._clamp_score(trust_target - 10.0)

        trust_alpha = max(0.03, alpha * 0.80)
        new_trust = cls._clamp_score(
            old_trust + trust_alpha * (trust_target - old_trust)
        )

        agent.reliability_score = new_reliability
        agent.trust_score = new_trust

        if str(agent_run.status).lower() == "completed":
            latency = max(int(agent_run.latency_ms or 0), 0)
            if previous_runs <= 0:
                agent.average_latency_ms = latency
            else:
                agent.average_latency_ms = int(
                    round((old_latency * previous_runs + latency) / (previous_runs + 1))
                )
            agent.completed_runs = previous_runs + 1

        return {
            "execution_outcome": execution_outcome,
            "old_reliability": round(old_reliability, 2),
            "new_reliability": new_reliability,
            "old_trust": round(old_trust, 2),
            "new_trust": new_trust,
            "hallucination_rate": round(old_hallucination, 2),
        }

    @classmethod
    def apply_verification_feedback(
        cls,
        database: Session,
        *,
        state: dict[str, Any],
        verification_result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Update claim-producing agents from deterministic verification results.

        The current complaint workflow attributes root-cause claims to
        ``root_cause_analysis``. A future test/adversarial agent can set
        ``source_agent_code`` on an injected finding and automatically receive
        the same evidence-based trust, reliability and hallucination feedback.
        """
        run_id = cls._uuid(state["workflow_run_id"])
        mission_id = cls._uuid(state["mission_id"])

        payload = verification_result or {}
        if isinstance(payload.get("output"), dict):
            payload = payload["output"]
        if isinstance(payload.get("verification"), dict):
            payload = payload["verification"]

        claims = payload.get("claims", []) if isinstance(payload, dict) else []
        if not isinstance(claims, list):
            return []

        claims_by_agent: dict[str, list[dict[str, Any]]] = {}
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            source_agent_code = str(
                claim.get("source_agent_code") or "root_cause_analysis"
            ).strip()
            if not source_agent_code:
                source_agent_code = "root_cause_analysis"
            claims_by_agent.setdefault(source_agent_code, []).append(claim)

        updates: list[dict[str, Any]] = []
        for source_agent_code, source_claims in claims_by_agent.items():
            agent = database.scalar(
                select(Agent).where(Agent.code == source_agent_code)
            )
            if agent is None or not source_claims:
                continue

            source_run = database.scalar(
                select(AgentRun)
                .where(
                    AgentRun.workflow_run_id == run_id,
                    AgentRun.agent_code == source_agent_code,
                )
                .order_by(AgentRun.attempt.desc(), AgentRun.created_at.desc())
            )
            if source_run is None:
                continue

            run_output = dict(source_run.output or {})
            feedback_marker = run_output.get("_cortex_verification_feedback")
            if isinstance(feedback_marker, dict) and feedback_marker.get("applied"):
                continue

            outcome_values = {
                "verified": 100.0,
                "needs_review": 60.0,
                "rejected": 10.0,
            }
            outcomes = [
                outcome_values.get(str(item.get("verification_result", "")).lower(), 40.0)
                for item in source_claims
            ]
            groundedness_values = [
                min(max(float(item.get("groundedness", 0.0) or 0.0), 0.0), 1.0)
                for item in source_claims
            ]

            claim_count = max(len(source_claims), 1)
            rejected_count = sum(
                str(item.get("verification_result", "")).lower() == "rejected"
                for item in source_claims
            )
            review_count = sum(
                str(item.get("verification_result", "")).lower() == "needs_review"
                for item in source_claims
            )
            contradiction_count = sum(
                bool(item.get("contradiction_found", False))
                for item in source_claims
            )

            outcome_score = sum(outcomes) / claim_count
            groundedness_score = sum(groundedness_values) / claim_count * 100.0
            observed_hallucination = cls._clamp_score(
                rejected_count / claim_count * 100.0
                + review_count / claim_count * 25.0
                + contradiction_count / claim_count * 15.0
            )
            claim_quality = cls._clamp_score(
                groundedness_score * 0.50
                + outcome_score * 0.35
                + (100.0 - observed_hallucination) * 0.15
            )

            previous_runs = int(agent.completed_runs or 0)
            alpha = cls._verification_learning_rate(previous_runs)
            old_reliability = float(agent.reliability_score or 80.0)
            old_trust = float(agent.trust_score or 80.0)
            old_hallucination = float(agent.hallucination_rate or 0.0)

            new_reliability = cls._clamp_score(
                old_reliability + (alpha * 0.65) * (outcome_score - old_reliability)
            )
            trust_target = cls._clamp_score(
                new_reliability * 0.55 + claim_quality * 0.45
            )
            new_trust = cls._clamp_score(
                old_trust + alpha * (trust_target - old_trust)
            )
            new_hallucination = cls._clamp_score(
                old_hallucination
                + alpha * (observed_hallucination - old_hallucination)
            )

            agent.reliability_score = new_reliability
            agent.trust_score = new_trust
            agent.hallucination_rate = new_hallucination

            feedback = {
                "applied": True,
                "claim_count": claim_count,
                "verified_count": sum(
                    str(item.get("verification_result", "")).lower() == "verified"
                    for item in source_claims
                ),
                "review_count": review_count,
                "rejected_count": rejected_count,
                "groundedness_score": round(groundedness_score, 2),
                "observed_hallucination": observed_hallucination,
                "claim_quality": claim_quality,
                "old_reliability": round(old_reliability, 2),
                "new_reliability": new_reliability,
                "old_trust": round(old_trust, 2),
                "new_trust": new_trust,
                "old_hallucination_rate": round(old_hallucination, 2),
                "new_hallucination_rate": new_hallucination,
            }
            run_output["_cortex_verification_feedback"] = feedback
            source_run.output = run_output

            AuditService.write(
                database,
                action="agent.verification_feedback",
                mission_id=mission_id,
                task_id=source_run.task_id,
                entity_type="agent",
                entity_id=str(agent.id),
                details={
                    "workflow_run_id": str(run_id),
                    "agent_code": source_agent_code,
                    **feedback,
                },
            )
            updates.append({"agent_code": source_agent_code, **feedback})

        return updates

    @classmethod
    def create_run(
        cls,
        database: Session,
        *,
        mission: Mission,
        initial_state: dict[str, Any],
        actor_user_id: UUID,
    ) -> WorkflowRun:
        run = WorkflowRun(
            mission_id=mission.id,
            status="queued",
            current_stage="queued",
            retry_count=0,
            state_snapshot=initial_state,
            started_at=None,
            completed_at=None,
        )
        database.add(run)
        mission.status = "queued"
        database.flush()
        AuditService.write(
            database,
            action="workflow.queued",
            actor_user_id=actor_user_id,
            mission_id=mission.id,
            entity_type="workflow_run",
            entity_id=str(run.id),
            details={"status": "queued"},
        )
        return run

    @classmethod
    def update_run(
        cls,
        database: Session,
        *,
        run_id: str | UUID,
        status: str | None = None,
        current_stage: str | None = None,
        retry_count: int | None = None,
        state_snapshot: dict[str, Any] | None = None,
        completed: bool = False,
    ) -> WorkflowRun:
        run = database.get(WorkflowRun, cls._uuid(run_id))
        if run is None:
            raise RuntimeError(f"Workflow run not found: {run_id}")
        if status is not None:
            run.status = status
        if current_stage is not None:
            run.current_stage = current_stage
        if retry_count is not None:
            run.retry_count = retry_count
        if state_snapshot is not None:
            run.state_snapshot = state_snapshot
        if run.started_at is None and status in {
            "running",
            "awaiting_approval",
            "completed",
            "failed",
        }:
            run.started_at = datetime.now(timezone.utc)
        if completed:
            run.completed_at = datetime.now(timezone.utc)
        return run

    @classmethod
    def record_agent_result(
        cls,
        database: Session,
        *,
        state: dict[str, Any],
        stage: str,
        result: dict[str, Any],
        attempt: int = 1,
    ) -> AgentRun:
        run_id = cls._uuid(state["workflow_run_id"])
        mission_id = cls._uuid(state["mission_id"])
        attempt = max(int(attempt), 1)
        agent_code = str(result.get("agent_code") or stage)

        existing = database.scalar(
            select(AgentRun).where(
                AgentRun.workflow_run_id == run_id,
                AgentRun.stage == stage,
                AgentRun.attempt == attempt,
            )
        )
        is_new_result = existing is None
        agent = database.scalar(select(Agent).where(Agent.code == agent_code))
        task_type = STAGE_TO_TASK_TYPE.get(stage)
        task = None
        if task_type:
            task = database.scalar(
                select(Task).where(
                    Task.mission_id == mission_id,
                    Task.task_type == task_type,
                )
            )

        if existing is None:
            existing = AgentRun(
                workflow_run_id=run_id,
                task_id=task.id if task else None,
                agent_id=agent.id if agent else None,
                agent_code=agent_code,
                stage=stage,
                attempt=attempt,
                status=str(result.get("status", "failed")),
                started_at=datetime.now(timezone.utc),
            )
            database.add(existing)

        existing.status = str(result.get("status", "failed"))
        existing.confidence = float(result.get("confidence", 0.0) or 0.0)
        existing.latency_ms = int(result.get("latency_ms", 0) or 0)
        existing.input_tokens = int(result.get("input_tokens", 0) or 0)
        existing.output_tokens = int(result.get("output_tokens", 0) or 0)
        existing.total_tokens = int(result.get("total_tokens", 0) or 0)
        existing.estimated_cost = float(result.get("estimated_cost", 0.0) or 0.0)
        existing.output = result.get("output") or {}
        existing.warnings = result.get("warnings") or []
        existing.errors = result.get("errors") or []
        existing.completed_at = datetime.now(timezone.utc)

        if task is not None:
            task.status = "completed" if existing.status == "completed" else "failed"

        performance_update: dict[str, float] | None = None
        if agent is not None and is_new_result:
            performance_update = cls._apply_execution_performance(agent, existing)

        AuditService.write(
            database,
            action=f"agent.{existing.status}",
            mission_id=mission_id,
            task_id=task.id if task else None,
            entity_type="agent_run",
            entity_id=str(existing.id),
            details={
                "workflow_run_id": str(run_id),
                "stage": stage,
                "agent_code": agent_code,
                "attempt": attempt,
                "latency_ms": existing.latency_ms,
                "total_tokens": existing.total_tokens,
                "confidence": existing.confidence,
                "performance_update": performance_update,
            },
        )
        return existing

    @classmethod
    def mark_task_types_skipped(
        cls,
        database: Session,
        *,
        mission_id: str | UUID,
        task_types: list[str],
    ) -> None:
        """Persist safe-stop semantics into the Mission Plan task rows."""
        if not task_types:
            return
        mission_uuid = cls._uuid(mission_id)
        tasks = list(
            database.scalars(
                select(Task).where(
                    Task.mission_id == mission_uuid,
                    Task.task_type.in_(task_types),
                )
            )
        )
        for task in tasks:
            if str(task.status).lower() not in {"completed", "failed"}:
                task.status = "skipped"

    @classmethod
    def sync_plan_and_routing(
        cls,
        database: Session,
        *,
        mission_id: str | UUID,
        tasks: list[dict[str, Any]],
        routing: list[dict[str, Any]],
    ) -> None:
        mission_uuid = cls._uuid(mission_id)
        agents_by_code = {
            agent.code: agent
            for agent in database.scalars(
                select(Agent).where(Agent.status.in_(["active", "test_only"]))
            )
        }
        route_by_task = {item["task_code"]: item for item in routing}
        existing_tasks = list(
            database.scalars(
                select(Task)
                .where(Task.mission_id == mission_uuid)
                .order_by(Task.sequence_order, Task.created_at)
            )
        )
        by_type = {task.task_type: task for task in existing_tasks}
        by_code: dict[str, Task] = {}

        for planned in tasks:
            task_type = str(planned["task_type"])
            task = by_type.get(task_type)
            route = route_by_task.get(str(planned["code"]), {})
            agent = agents_by_code.get(str(route.get("selected_agent_code", "")))
            if task is None:
                task = Task(
                    mission_id=mission_uuid,
                    title=str(planned["title"]),
                    description=str(planned["description"]),
                    task_type=task_type,
                    sequence_order=int(planned["stage"]),
                    execution_mode=str(planned["execution_mode"]),
                    priority=str(planned["priority"]),
                    risk_level=str(planned["risk_level"]),
                    status="waiting",
                    expected_output=str(planned["expected_output"]),
                    assigned_agent_id=agent.id if agent else None,
                )
                database.add(task)
                database.flush()
                by_type[task_type] = task
            else:
                task.title = str(planned["title"])
                task.description = str(planned["description"])
                task.sequence_order = int(planned["stage"])
                task.execution_mode = str(planned["execution_mode"])
                task.priority = str(planned["priority"])
                task.risk_level = str(planned["risk_level"])
                task.expected_output = str(planned["expected_output"])
                task.assigned_agent_id = agent.id if agent else task.assigned_agent_id
            by_code[str(planned["code"])] = task

        # The mission may have been created with a generic complaint-oriented template.
        # Once the live planner identifies a logistics profile, remove task rows that do
        # not belong to the current workflow so Mission Plan shows only relevant agents.
        planned_types = {str(item["task_type"]) for item in tasks}
        stale_tasks = [task for task in existing_tasks if task.task_type not in planned_types]
        stale_ids = [task.id for task in stale_tasks]
        if stale_ids:
            database.execute(
                delete(TaskDependency).where(TaskDependency.task_id.in_(stale_ids))
            )
            database.execute(
                delete(TaskDependency).where(TaskDependency.depends_on_task_id.in_(stale_ids))
            )
            database.execute(delete(Task).where(Task.id.in_(stale_ids)))

        current_ids = [task.id for task in by_code.values()]
        if current_ids:
            database.execute(
                delete(TaskDependency).where(TaskDependency.task_id.in_(current_ids))
            )
        for planned in tasks:
            task = by_code.get(str(planned["code"]))
            if task is None:
                continue
            for dependency_code in planned.get("depends_on", []):
                dependency = by_code.get(str(dependency_code))
                if dependency is not None and dependency.id != task.id:
                    database.add(
                        TaskDependency(
                            task_id=task.id,
                            depends_on_task_id=dependency.id,
                        )
                    )

        mission = database.get(Mission, mission_uuid)
        if mission is not None:
            mission.plan_source = "azure" if any(
                item.get("score", 0) for item in routing
            ) else "template"
            mission.status = "running"

    @classmethod
    def persist_trust_graph(
        cls,
        database: Session,
        *,
        state: dict[str, Any],
        trust_graph: dict[str, Any],
    ) -> None:
        run_id = cls._uuid(state["workflow_run_id"])
        mission_id = cls._uuid(state["mission_id"])
        database.execute(delete(ClaimRecord).where(ClaimRecord.workflow_run_id == run_id))
        database.execute(
            delete(EvidenceRecord).where(EvidenceRecord.workflow_run_id == run_id)
        )

        for row in trust_graph.get("claim_evidence_table", []):
            database.add(
                ClaimRecord(
                    workflow_run_id=run_id,
                    mission_id=mission_id,
                    external_claim_id=str(row.get("claim_id", "")),
                    category=str(row.get("category", "Uncategorised")),
                    claim_text=str(row.get("claim", "")),
                    verification_status=str(
                        row.get("verification_result", "needs_review")
                    ),
                    groundedness=float(row.get("groundedness", 0.0) or 0.0),
                    hallucination_risk=float(
                        row.get("hallucination_risk", 1.0) or 1.0
                    ),
                    contradiction_found=bool(row.get("contradiction_found", False)),
                    citation_status=str(row.get("citation_status", "unknown")),
                    evidence_ids=row.get("evidence_ids", []) or [],
                    metadata_json=row,
                )
            )

        for node in trust_graph.get("nodes", []):
            if node.get("node_type") != "evidence":
                continue
            metadata = node.get("metadata") or {}
            database.add(
                EvidenceRecord(
                    workflow_run_id=run_id,
                    mission_id=mission_id,
                    external_evidence_id=str(node.get("node_id", "")),
                    evidence_type=str(metadata.get("evidence_type", "supporting")),
                    source_title=str(node.get("label", ""))[:255],
                    source_path=metadata.get("source_path"),
                    text_excerpt=str(node.get("label", ""))[:2000],
                    strength=str(node.get("status", "supporting")),
                    metadata_json=metadata,
                )
            )

    @classmethod
    def persist_governance(
        cls,
        database: Session,
        *,
        state: dict[str, Any],
        governance: dict[str, Any],
    ) -> GovernanceDecision:
        run_id = cls._uuid(state["workflow_run_id"])
        mission_id = cls._uuid(state["mission_id"])
        record = database.scalar(
            select(GovernanceDecision).where(
                GovernanceDecision.workflow_run_id == run_id
            )
        )
        if record is None:
            record = GovernanceDecision(
                workflow_run_id=run_id,
                mission_id=mission_id,
                decision=str(governance.get("decision", "block")),
                risk_level=str(governance.get("risk_level", "prohibited")),
                approval_required=bool(governance.get("approval_required", False)),
            )
            database.add(record)
        record.decision = str(governance.get("decision", "block"))
        record.risk_level = str(governance.get("risk_level", "prohibited"))
        record.approval_required = bool(governance.get("approval_required", False))
        record.reason = governance.get("approval_reason")
        record.policy_checks = governance.get("policy_checks", []) or []
        record.permitted_claim_ids = governance.get("permitted_claim_ids", []) or []
        record.excluded_claim_ids = governance.get("excluded_claim_ids", []) or []
        record.metadata_json = governance
        return record

    @classmethod
    def persist_evaluation(
        cls,
        database: Session,
        *,
        state: dict[str, Any],
        evaluation: dict[str, Any],
    ) -> EvaluationResult:
        run_id = cls._uuid(state["workflow_run_id"])
        mission_id = cls._uuid(state["mission_id"])
        record = database.scalar(
            select(EvaluationResult).where(EvaluationResult.workflow_run_id == run_id)
        )
        if record is None:
            record = EvaluationResult(
                workflow_run_id=run_id,
                mission_id=mission_id,
                mission_success_score=float(
                    evaluation.get("mission_success_score", 0.0) or 0.0
                ),
                status=str(evaluation.get("status", "Failed")),
                component_scores=evaluation.get("component_scores", {}) or {},
                weakest_component=str(
                    evaluation.get("weakest_component", "unknown")
                ),
                recommended_improvement=str(
                    evaluation.get("recommended_improvement", "")
                ),
            )
            database.add(record)
        record.mission_success_score = float(
            evaluation.get("mission_success_score", 0.0) or 0.0
        )
        record.status = str(evaluation.get("status", "Failed"))
        record.component_scores = evaluation.get("component_scores", {}) or {}
        record.weakest_component = str(evaluation.get("weakest_component", "unknown"))
        record.recommended_improvement = str(
            evaluation.get("recommended_improvement", "")
        )
        record.metrics = evaluation
        return record

    @classmethod
    def ensure_approval(
        cls,
        database: Session,
        *,
        state: dict[str, Any],
        governance: dict[str, Any],
    ) -> WorkflowApproval:
        run_id = cls._uuid(state["workflow_run_id"])
        existing = database.scalar(
            select(WorkflowApproval).where(WorkflowApproval.workflow_run_id == run_id)
        )
        if existing is not None:
            if (
                state.get("approval_status") == "request_revision"
                and existing.status != "pending"
            ):
                existing.status = "pending"
                existing.reviewer_id = None
                existing.review_comment = None
                existing.reviewed_at = None
                existing.reason = str(
                    governance.get("approval_reason")
                    or "Revised recommendation requires human approval."
                )
                existing.action_payload = governance.get("approval_payload") or {}
                existing.risk_level = str(governance.get("risk_level", "high"))
            return existing
        record = WorkflowApproval(
            workflow_run_id=run_id,
            mission_id=cls._uuid(state["mission_id"]),
            reason=str(
                governance.get("approval_reason")
                or "High-impact recommendation requires human approval."
            ),
            action_payload=governance.get("approval_payload") or {},
            risk_level=str(governance.get("risk_level", "high")),
            status="pending",
            requested_by_agent="guardian_governance",
        )
        database.add(record)
        database.flush()
        return record

    @classmethod
    def resolve_approval(
        cls,
        database: Session,
        *,
        approval_id: str | UUID,
        reviewer_id: UUID,
        action: str,
        comment: str,
    ) -> WorkflowApproval:
        record = database.get(WorkflowApproval, cls._uuid(approval_id))
        if record is None:
            raise RuntimeError("Approval request not found.")
        record.status = action
        record.reviewer_id = reviewer_id
        record.review_comment = comment
        record.reviewed_at = datetime.now(timezone.utc)
        AuditService.write(
            database,
            action=f"approval.{action}",
            actor_user_id=reviewer_id,
            mission_id=record.mission_id,
            entity_type="workflow_approval",
            entity_id=str(record.id),
            details={"workflow_run_id": str(record.workflow_run_id), "comment": comment},
        )
        return record
