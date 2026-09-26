from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.agents.base_agent import AgentExecutionResult, BaseAgent
from backend.services.operational_evidence_processor import (
    OperationalEvidenceProcessor,
)


RootCauseStatus = Literal["confirmed", "probable", "unverified"]
RiskLevel = Literal["low", "medium", "high", "critical"]


class RootCauseFinding(BaseModel):
    title: str
    category: str
    status: RootCauseStatus
    root_cause: str
    reasoning: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    complaint_evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    business_impact: str
    recommended_action: str
    risk_level: RiskLevel
    human_approval_required: bool = False
    evidence_gaps: list[str] = Field(default_factory=list)


class RootCauseAnalysisOutput(BaseModel):
    executive_summary: str
    findings: list[RootCauseFinding]
    cross_cutting_risks: list[str]
    recommended_next_checks: list[str]
    limitations: list[str]


class RootCauseAnalysisAgent(BaseAgent):
    code = "root_cause_analysis"
    name = "Root-Cause Analysis Agent"

    def __init__(
        self,
        processor: OperationalEvidenceProcessor | None = None,
    ) -> None:
        super().__init__()
        self.processor = processor or OperationalEvidenceProcessor()

    @staticmethod
    def _complaint_sections(
        complaint_output: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        analysis = complaint_output.get("analysis", complaint_output)
        dataset_summary = complaint_output.get("dataset_summary", {})

        if not isinstance(analysis, dict):
            analysis = {}
        if not isinstance(dataset_summary, dict):
            dataset_summary = {}

        return analysis, dataset_summary

    @classmethod
    def _complaint_ids_by_category(
        cls,
        complaint_output: dict[str, Any],
    ) -> dict[str, list[str]]:
        analysis, _ = cls._complaint_sections(complaint_output)
        index: dict[str, list[str]] = {}

        for section_name in ("recurring_issues", "emerging_issues"):
            for item in analysis.get(section_name, []) or []:
                if not isinstance(item, dict):
                    continue
                category = str(item.get("category", "Other"))
                ids = item.get("evidence_complaint_ids", []) or []
                bucket = index.setdefault(category, [])
                bucket.extend(str(value) for value in ids if str(value).strip())

        return {
            category: sorted(set(values))[:8]
            for category, values in index.items()
        }

    @classmethod
    def _category_impact(
        cls,
        complaint_output: dict[str, Any],
    ) -> dict[str, str]:
        analysis, dataset_summary = cls._complaint_sections(complaint_output)
        impact: dict[str, str] = {}

        for item in analysis.get("recurring_issues", []) or []:
            if not isinstance(item, dict):
                continue
            category = str(item.get("category", "Other"))
            business_impact = str(item.get("business_impact", "")).strip()
            if business_impact:
                impact[category] = business_impact

        for metric in dataset_summary.get("category_metrics", []) or []:
            if not isinstance(metric, dict):
                continue
            category = str(metric.get("category", "Other"))
            if category in impact:
                continue
            impact[category] = (
                f"{metric.get('count', 0)} complaints; "
                f"SLA breach rate {metric.get('sla_breach_rate', 0)}%; "
                f"escalation rate {metric.get('escalation_rate', 0)}%."
            )

        return impact

    @staticmethod
    def _compact_complaint_context(
        complaint_output: dict[str, Any],
    ) -> dict[str, Any]:
        analysis = complaint_output.get("analysis", complaint_output)
        summary = complaint_output.get("dataset_summary", {})

        if not isinstance(analysis, dict):
            analysis = {}
        if not isinstance(summary, dict):
            summary = {}

        return {
            "executive_summary": analysis.get("executive_summary", ""),
            "recurring_issues": analysis.get("recurring_issues", []),
            "emerging_issues": analysis.get("emerging_issues", []),
            "next_analysis_questions": analysis.get(
                "next_analysis_questions", []
            ),
            "category_metrics": summary.get("category_metrics", []),
            "overall_metrics": summary.get("overall_metrics", {}),
            "decision_signals": summary.get("decision_signals", {}),
            "synthetic_data": bool(
                complaint_output.get("synthetic_data", True)
            ),
        }

    @staticmethod
    def _compact_operational_context(
        operational_summary: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "incident_evidence": operational_summary.get(
                "incident_evidence", []
            ),
            "release_evidence": operational_summary.get(
                "release_evidence", []
            ),
            "transaction_signals": operational_summary.get(
                "transaction_signals", []
            ),
            "support_signals": operational_summary.get(
                "support_signals", []
            ),
            "category_evidence_index": operational_summary.get(
                "category_evidence_index", {}
            ),
            "limitations": operational_summary.get("limitations", []),
        }

    @classmethod
    def _build_mock_output(
        cls,
        complaint_output: dict[str, Any],
        operational_summary: dict[str, Any],
    ) -> RootCauseAnalysisOutput:
        complaint_ids = cls._complaint_ids_by_category(complaint_output)
        impact = cls._category_impact(complaint_output)
        category_index = operational_summary.get("category_evidence_index", {})
        incidents = operational_summary.get("incident_evidence", [])

        incident_by_category = {
            str(item.get("category")): item
            for item in incidents
            if isinstance(item, dict)
        }

        findings: list[RootCauseFinding] = []

        definitions = [
            (
                "Payment Failure",
                "Payment gateway timeout misconfiguration",
                (
                    "A production timeout setting introduced by REL-6.4-PAY "
                    "caused valid mobile payment requests to expire."
                ),
                "confirmed",
                0.96,
                "high",
                (
                    "Keep the corrected timeout configuration, run payment "
                    "regression tests, and require approval before any rollback."
                ),
            ),
            (
                "Login & Access",
                "Authentication token-cache regression",
                (
                    "Token-cache invalidation changes in REL-6.4-AUTH caused "
                    "refresh failures and repeated sign-in loops."
                ),
                "confirmed",
                0.94,
                "medium",
                (
                    "Retain the corrected authentication build and add automated "
                    "token-refresh regression tests."
                ),
            ),
            (
                "Refund Delay",
                "Regional refund-processing capacity shortage",
                (
                    "A growing manual reconciliation queue combined with lower "
                    "available staffing in South and West is the most likely "
                    "driver of refund delays."
                ),
                "probable",
                0.83,
                "medium",
                (
                    "Temporarily rebalance refund work across regions, restore "
                    "staffing capacity, and measure queue age daily."
                ),
            ),
            (
                "Delivery Tracking",
                "Courier synchronization scheduler delay",
                (
                    "A scheduler and queue-batching change is temporally aligned "
                    "with stale courier-status updates, but the incident remains "
                    "under investigation."
                ),
                "probable",
                0.78,
                "high",
                (
                    "Compare scheduler and partner API logs, test a controlled "
                    "configuration rollback, and obtain approval before production "
                    "rollback."
                ),
            ),
        ]

        for (
            category,
            title,
            root_cause,
            status,
            confidence,
            risk,
            action,
        ) in definitions:
            evidence_bucket = category_index.get(category, {})
            evidence_ids = list(evidence_bucket.get("evidence_ids", []))
            direct_ids = list(evidence_bucket.get("direct_evidence_ids", []))
            incident = incident_by_category.get(category, {})

            reasoning = (
                f"Complaint patterns for {category} overlap with "
                f"{len(evidence_ids)} operational evidence item(s)."
            )
            confirmed_cause = str(
                incident.get("confirmed_root_cause", "")
            ).strip()
            if confirmed_cause:
                reasoning += f" The incident record states: {confirmed_cause}"

            gaps: list[str] = []
            if status == "probable":
                gaps.append(
                    "No closed incident record contains a directly confirmed root cause."
                )
            if status == "confirmed" and not direct_ids:
                status = "probable"
                confidence = min(confidence, 0.88)
                gaps.append(
                    "The expected direct evidence was unavailable, so the finding "
                    "was downgraded."
                )

            findings.append(
                RootCauseFinding(
                    title=title,
                    category=category,
                    status=status,
                    root_cause=root_cause,
                    reasoning=reasoning,
                    supporting_evidence_ids=evidence_ids,
                    complaint_evidence_ids=complaint_ids.get(category, []),
                    confidence=confidence,
                    business_impact=impact.get(
                        category,
                        "The issue creates service, SLA, and customer-experience risk.",
                    ),
                    recommended_action=action,
                    risk_level=risk,
                    human_approval_required=(
                        risk in {"high", "critical"}
                        or "rollback" in action.lower()
                    ),
                    evidence_gaps=gaps,
                )
            )

        return RootCauseAnalysisOutput(
            executive_summary=(
                "Operational evidence confirms payment and authentication causes, "
                "while refund and delivery causes remain probable pending stronger "
                "direct evidence. High-impact production actions require human approval."
            ),
            findings=findings,
            cross_cutting_risks=[
                "Multiple issues followed production or workflow changes without "
                "sufficient post-release monitoring.",
                "SLA pressure and repeated contacts amplify customer impact during "
                "operational incidents.",
            ],
            recommended_next_checks=[
                "Verify payment timeout settings against the approved configuration baseline.",
                "Review authentication regression-test coverage for token refresh flows.",
                "Confirm refund queue age, staffing capacity, and reconciliation throughput.",
                "Compare courier partner logs with scheduler execution records.",
            ],
            limitations=list(operational_summary.get("limitations", [])),
        )

    @staticmethod
    def _apply_evidence_guardrails(
        output: RootCauseAnalysisOutput,
        operational_summary: dict[str, Any],
    ) -> list[str]:
        warnings: list[str] = []
        evidence_catalog = operational_summary.get("evidence_catalog", [])
        evidence_map = {
            str(item.get("evidence_id")): item
            for item in evidence_catalog
            if isinstance(item, dict) and item.get("evidence_id")
        }

        high_impact_terms = {
            "rollback",
            "shutdown",
            "disable service",
            "revoke access",
            "production change",
        }

        for finding in output.findings:
            original_ids = list(finding.supporting_evidence_ids)
            finding.supporting_evidence_ids = sorted(
                {
                    evidence_id
                    for evidence_id in original_ids
                    if evidence_id in evidence_map
                }
            )

            removed_count = len(set(original_ids)) - len(
                finding.supporting_evidence_ids
            )
            if removed_count > 0:
                warnings.append(
                    f"Removed {removed_count} unsupported evidence ID(s) from "
                    f"'{finding.title}'."
                )

            direct_ids = [
                evidence_id
                for evidence_id in finding.supporting_evidence_ids
                if evidence_map[evidence_id].get("strength") == "direct"
            ]

            if finding.status == "confirmed" and not direct_ids:
                finding.status = (
                    "probable"
                    if finding.supporting_evidence_ids
                    else "unverified"
                )
                finding.confidence = min(
                    finding.confidence,
                    0.88 if finding.supporting_evidence_ids else 0.55,
                )
                finding.evidence_gaps.append(
                    "Confirmed classification was downgraded because no direct "
                    "operational evidence was available."
                )
                warnings.append(
                    f"Downgraded '{finding.title}' because direct evidence was missing."
                )

            if not finding.supporting_evidence_ids:
                finding.status = "unverified"
                finding.confidence = min(finding.confidence, 0.55)
                if not finding.evidence_gaps:
                    finding.evidence_gaps.append(
                        "No valid operational evidence was linked to this finding."
                    )

            if finding.status == "probable":
                finding.confidence = min(finding.confidence, 0.89)
            elif finding.status == "unverified":
                finding.confidence = min(finding.confidence, 0.59)

            action_text = finding.recommended_action.lower()
            if any(term in action_text for term in high_impact_terms):
                finding.human_approval_required = True
                if finding.risk_level in {"low", "medium"}:
                    finding.risk_level = "high"

        return warnings

    def run(
        self,
        *,
        complaint_output: dict[str, Any],
        operations_directory: str | Path,
        knowledge_evidence: list[dict[str, Any]] | None = None,
    ) -> AgentExecutionResult:
        started_at = perf_counter()

        try:
            processed = self.processor.process_directory(operations_directory)
        except Exception as exc:
            return self.failure(
                error=f"Operational evidence processing failed: {exc}",
                latency_ms=int((perf_counter() - started_at) * 1000),
            )

        operational_summary = processed.summary
        mock_output = self._build_mock_output(
            complaint_output,
            operational_summary,
        )

        system_prompt = """
        You are the CORTEX Root-Cause Analysis Agent.

        Connect observed complaint patterns to the supplied synthetic operational
        evidence. Produce decision support, not unsupported speculation.

        Classification rules:
        - confirmed: at least one direct operational evidence item confirms the cause.
        - probable: multiple independent temporal, statistical, or corroborating
          evidence items support the cause, but direct confirmation is incomplete.
        - unverified: evidence is missing, weak, contradictory, or based only on
          complaint wording.

        Requirements:
        1. Use only evidence IDs included in the supplied evidence catalog.
        2. Do not invent complaint IDs, incident IDs, release IDs, counts, or dates.
        3. Explain why each cause has its classification.
        4. State evidence gaps explicitly.
        5. Mark rollback, shutdown, access revocation, and other high-impact
           production actions as requiring human approval.
        6. Treat all information as synthetic demonstration data.
        7. Keep the output concise and enterprise-friendly.
        """

        prompt_payload = {
            "complaint_intelligence": self._compact_complaint_context(
                complaint_output
            ),
            "operational_evidence": self._compact_operational_context(
                operational_summary
            ),
            "retrieved_policy_evidence": knowledge_evidence or [],
        }

        user_prompt = (
            "Analyse the following synthetic complaint intelligence and "
            "operational evidence. Return the required structured root-cause "
            "analysis.\n\n"
            + json.dumps(prompt_payload, ensure_ascii=False, indent=2)
        )

        llm_result = self.llm_service.invoke_structured(
            agent_name=self.name,
            schema=RootCauseAnalysisOutput,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            mock_value=mock_output,
        )

        total_latency_ms = int((perf_counter() - started_at) * 1000)

        if llm_result.status != "completed":
            return self.failure(
                error=f"Root-cause analysis failed: {llm_result.error}",
                latency_ms=total_latency_ms,
            )

        output = llm_result.content
        if not isinstance(output, RootCauseAnalysisOutput):
            try:
                output = RootCauseAnalysisOutput.model_validate(output)
            except Exception as exc:
                return self.failure(
                    error=f"Invalid root-cause output: {exc}",
                    latency_ms=total_latency_ms,
                )

        warnings = self._apply_evidence_guardrails(
            output,
            operational_summary,
        )

        confidence_values = [finding.confidence for finding in output.findings]
        overall_confidence = (
            mean(confidence_values) if confidence_values else 0.0
        )
        evidence_ids = sorted(
            {
                evidence_id
                for finding in output.findings
                for evidence_id in finding.supporting_evidence_ids
            }
        )

        return self.success(
            output={
                "analysis": output.model_dump(),
                "operational_evidence_summary": operational_summary,
                "provider": llm_result.provider,
                "deployment": llm_result.deployment,
                "synthetic_data": True,
            },
            confidence=round(overall_confidence, 4),
            latency_ms=total_latency_ms,
            usage=llm_result.usage,
            evidence_ids=evidence_ids,
            warnings=warnings,
        )
