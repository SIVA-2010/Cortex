from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.agents.report_agent import ExecutiveReportAgent
from backend.config import get_settings
from backend.models import AgentRun, ShadowRun, WorkflowRun
from backend.services.llm import LLMService


class ShadowBenchService:
    def __init__(self) -> None:
        self.settings = get_settings()

    @staticmethod
    def _report_payload(value: dict[str, Any] | None) -> dict[str, Any]:
        payload = value or {}
        output = payload.get("output", payload)
        return output.get("report", output) if isinstance(output, dict) else {}

    @staticmethod
    def _evidence_ids(report: dict[str, Any]) -> set[str]:
        return {
            str(evidence_id)
            for finding in report.get("key_findings", [])
            for evidence_id in finding.get("evidence_ids", [])
        }

    @staticmethod
    def _categories(report: dict[str, Any]) -> set[str]:
        return {
            str(finding.get("category", "")).strip().lower()
            for finding in report.get("key_findings", [])
            if str(finding.get("category", "")).strip()
        }

    def run_report_replay(
        self,
        database: Session,
        *,
        workflow_run_id: UUID,
    ) -> ShadowRun:
        workflow_run = database.get(WorkflowRun, workflow_run_id)
        if workflow_run is None:
            raise ValueError("Workflow run not found.")
        state = dict(workflow_run.state_snapshot or {})
        required = (
            "root_cause_result",
            "verification_result",
            "governance_result",
            "evaluation_result",
            "final_report",
        )
        missing = [key for key in required if not state.get(key)]
        if missing:
            raise ValueError(
                "The workflow is not ready for replay. Missing: " + ", ".join(missing)
            )

        shadow_deployment = self.settings.azure_openai_shadow_deployment.strip()
        if not shadow_deployment:
            if not self.settings.shadowbench_use_main_if_unset:
                raise ValueError(
                    "Configure AZURE_OPENAI_SHADOW_DEPLOYMENT before running ShadowBench."
                )
            shadow_deployment = self.settings.azure_openai_deployment

        production_agent_run = database.scalar(
            select(AgentRun)
            .where(
                AgentRun.workflow_run_id == workflow_run_id,
                AgentRun.stage == "generate_report",
            )
            .order_by(AgentRun.attempt.desc())
        )
        production_report = self._report_payload(state.get("final_report"))
        production_metrics = {
            "quality": 100.0,
            "groundedness": float(
                state.get("verification_result", {})
                .get("output", {})
                .get("verification", {})
                .get("overall_groundedness", 0.0)
            ),
            "hallucination_risk": float(
                state.get("verification_result", {})
                .get("output", {})
                .get("verification", {})
                .get("overall_hallucination_risk", 1.0)
            ),
            "latency_ms": int(production_agent_run.latency_ms if production_agent_run else 0),
            "total_tokens": int(production_agent_run.total_tokens if production_agent_run else 0),
            "estimated_cost": float(
                production_agent_run.estimated_cost if production_agent_run else 0.0
            ),
            "reliability": 1.0 if production_report else 0.0,
        }

        record = ShadowRun(
            workflow_run_id=workflow_run_id,
            production_agent_code="report_agent",
            shadow_agent_code="report_agent_shadow",
            production_deployment=self.settings.azure_openai_deployment,
            shadow_deployment=shadow_deployment,
            status="running",
            production_metrics=production_metrics,
            shadow_metrics={},
            recommendation="Evaluation in progress.",
        )
        database.add(record)
        database.flush()

        shadow_settings = self.settings.model_copy(
            update={"azure_openai_deployment": shadow_deployment}
        )
        shadow_agent = ExecutiveReportAgent(llm_service=LLMService(shadow_settings))
        result = shadow_agent.run(
            root_cause_result=state["root_cause_result"],
            verification_result=state["verification_result"],
            governance_result=state["governance_result"],
            evaluation_result=state["evaluation_result"],
            business_objective=state.get("mission", {}).get("objective", ""),
        )
        result_payload = result.model_dump(mode="json")
        candidate_report = self._report_payload(result_payload)

        baseline_evidence = self._evidence_ids(production_report)
        candidate_evidence = self._evidence_ids(candidate_report)
        baseline_categories = self._categories(production_report)
        candidate_categories = self._categories(candidate_report)
        evidence_precision = (
            len(candidate_evidence & baseline_evidence) / max(len(candidate_evidence), 1)
        )
        category_coverage = (
            len(candidate_categories & baseline_categories) / max(len(baseline_categories), 1)
        )
        quality = round((evidence_precision * 0.6 + category_coverage * 0.4) * 100, 2)
        shadow_metrics = {
            "quality": quality,
            "groundedness": round(evidence_precision, 4),
            "hallucination_risk": round(1.0 - evidence_precision, 4),
            "latency_ms": int(result.latency_ms),
            "total_tokens": int(result.total_tokens),
            "estimated_cost": float(result.estimated_cost),
            "reliability": 1.0 if result.status == "completed" else 0.0,
        }

        same_deployment = shadow_deployment == self.settings.azure_openai_deployment
        quality_pass = quality >= 95.0
        safety_pass = (
            shadow_metrics["hallucination_risk"]
            <= production_metrics["hallucination_risk"] + 0.01
        )
        efficiency_pass = (
            shadow_metrics["latency_ms"] < production_metrics["latency_ms"]
            or shadow_metrics["estimated_cost"] < production_metrics["estimated_cost"]
        )
        if result.status != "completed":
            recommendation = "Do not promote: the shadow execution failed."
            status = "failed"
        elif same_deployment:
            recommendation = (
                "Replay completed. Configure a distinct shadow deployment for a "
                "promotion decision."
            )
            status = "completed"
        elif quality_pass and safety_pass and efficiency_pass:
            recommendation = (
                "Promotion recommended: quality and governance were preserved while "
                "latency or cost improved."
            )
            status = "completed"
        else:
            recommendation = (
                "Do not promote yet: quality, safety or efficiency thresholds were not met."
            )
            status = "completed"

        record.status = status
        record.shadow_metrics = shadow_metrics
        record.recommendation = recommendation
        record.shadow_output = result_payload
        record.completed_at = datetime.now(timezone.utc)
        return record
