from __future__ import annotations

import json
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.agents.base_agent import AgentExecutionResult, BaseAgent


RecommendationStatus = Literal[
    "approved_for_use",
    "pending_human_approval",
    "additional_verification_required",
]


class ExecutiveFinding(BaseModel):
    claim_id: str
    category: str
    finding: str
    root_cause_status: str
    verification_status: str
    confidence: float = Field(ge=0.0, le=1.0)
    groundedness: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    business_impact: str


class ExecutiveRecommendation(BaseModel):
    priority: int = Field(ge=1)
    category: str
    action: str
    rationale: str
    risk_level: str
    approval_required: bool
    status: RecommendationStatus


class ExecutiveReportOutput(BaseModel):
    title: str
    synthetic_data_notice: str
    executive_summary: str
    business_objective: str
    key_findings: list[ExecutiveFinding]
    risks: list[str]
    recommendations: list[ExecutiveRecommendation]
    governance_status: str
    mission_success_score: float = Field(ge=0.0, le=100.0)
    mission_status: str
    next_recommended_action: str
    limitations: list[str] = Field(default_factory=list)


class ExecutiveReportAgent(BaseAgent):
    code = "report_agent"
    name = "Executive Report Agent"

    @staticmethod
    def _payload(value: dict[str, Any], key: str) -> dict[str, Any]:
        payload = value
        if isinstance(payload.get("output"), dict):
            payload = payload["output"]
        if isinstance(payload.get(key), dict):
            payload = payload[key]
        return payload

    @classmethod
    def _build_baseline(
        cls,
        *,
        root_cause_result: dict[str, Any],
        verification_result: dict[str, Any],
        governance_result: dict[str, Any],
        evaluation_result: dict[str, Any],
        business_objective: str,
        workflow_profile: str = "complaint",
    ) -> ExecutiveReportOutput:
        root_payload = root_cause_result
        if isinstance(root_payload.get("output"), dict):
            root_payload = root_payload["output"]
        analysis = root_payload.get("analysis", root_payload)
        findings = analysis.get("findings", []) if isinstance(analysis, dict) else []
        if not isinstance(findings, list):
            findings = []

        verification = cls._payload(verification_result, "verification")
        governance = cls._payload(governance_result, "governance")
        evaluation = cls._payload(evaluation_result, "evaluation")

        claims = verification.get("claims", [])
        if not isinstance(claims, list):
            claims = []

        finding_by_id: dict[str, dict[str, Any]] = {}
        for index, finding in enumerate(findings, start=1):
            if not isinstance(finding, dict):
                continue
            finding_id = str(finding.get("finding_id") or f"RC-{index:03d}")
            finding_by_id[finding_id] = finding

        permitted = set(governance.get("permitted_claim_ids", []))
        report_findings: list[ExecutiveFinding] = []
        recommendations: list[ExecutiveRecommendation] = []

        for claim in claims:
            if not isinstance(claim, dict):
                continue
            claim_id = str(claim.get("claim_id", ""))
            if claim_id not in permitted:
                continue

            finding = finding_by_id.get(str(claim.get("finding_id", "")), {})
            category = str(claim.get("category", "Uncategorised"))
            business_impact = str(
                finding.get("business_impact", "Business impact requires confirmation.")
            )
            report_findings.append(
                ExecutiveFinding(
                    claim_id=claim_id,
                    category=category,
                    finding=str(claim.get("claim", "")),
                    root_cause_status=str(claim.get("root_cause_status", "unverified")),
                    verification_status=str(claim.get("verification_result", "unknown")),
                    confidence=float(finding.get("confidence", 0.0) or 0.0),
                    groundedness=float(claim.get("groundedness", 0.0) or 0.0),
                    evidence_ids=[str(value) for value in claim.get("evidence_ids", [])],
                    business_impact=business_impact,
                )
            )

            action = str(finding.get("recommended_action", "")).strip()
            if action:
                approval_required = bool(
                    finding.get("human_approval_required", False)
                    or str(finding.get("approval_level", "")).lower()
                    in {"reviewer", "administrator", "admin"}
                    or governance.get("approval_required", False)
                )
                if approval_required:
                    status: RecommendationStatus = "pending_human_approval"
                elif governance.get("decision") == "verify":
                    status = "additional_verification_required"
                else:
                    status = "approved_for_use"

                recommendations.append(
                    ExecutiveRecommendation(
                        priority=len(recommendations) + 1,
                        category=category,
                        action=action,
                        rationale=str(
                            finding.get("reasoning")
                            or finding.get("evidence_summary")
                            or "The recommendation follows the verified evidence."
                        ),
                        risk_level=str(finding.get("risk_level", "medium")),
                        approval_required=approval_required,
                        status=status,
                    )
                )

        recommendations = sorted(
            recommendations,
            key=lambda item: (
                not item.approval_required,
                -max(
                    [
                        finding.confidence
                        for finding in report_findings
                        if finding.category == item.category
                    ]
                    or [0.0]
                ),
            ),
        )
        for index, item in enumerate(recommendations, start=1):
            item.priority = index

        risks = list(analysis.get("cross_cutting_risks", [])) if isinstance(analysis, dict) else []
        if not risks and isinstance(analysis, dict):
            risks = list(analysis.get("risks", []))
        risks = [str(value) for value in risks]

        decision = str(governance.get("decision", "block"))
        score = float(evaluation.get("mission_success_score", 0.0) or 0.0)
        mission_status = str(evaluation.get("status", "Failed"))

        if decision == "approval_required":
            next_action = "A reviewer must approve or reject the pending high-impact recommendation."
        elif decision == "verify":
            next_action = "Complete the additional verification checks before using the recommendations."
        elif decision == "block":
            next_action = "Resolve governance failures and rerun the mission."
        elif recommendations:
            next_action = recommendations[0].action
        else:
            next_action = "No approved recommendation is available."

        profile = str(workflow_profile or "complaint").lower()
        if profile == "logistics":
            title = "CORTEX Logistics Fulfilment Decision Report"
            synthetic_notice = (
                "Synthetic logistics demonstration data — no real carrier booking is performed."
            )
            executive_summary = (
                f"CORTEX produced {len(report_findings)} verified logistics fulfilment finding(s). "
                f"The governance decision is {decision}, and the Mission Success Score "
                f"is {score:.2f}/100."
            )
            limitations = [
                "Orders, products and carrier records are synthetic demonstration data.",
                "Tracking and fulfilment actions are simulated; no external carrier system is called.",
                "Rejected claims are excluded from the executive findings.",
            ]
        else:
            title = "CORTEX Customer Complaint Intelligence Decision Report"
            synthetic_notice = (
                "Synthetic demonstration data — not real customer or enterprise facts."
            )
            executive_summary = (
                f"CORTEX identified {len(report_findings)} verified root-cause finding(s). "
                f"The governance decision is {decision}, and the Mission Success Score "
                f"is {score:.2f}/100."
            )
            limitations = [
                "All complaints and operational records are synthetic demonstration data.",
                "Rejected claims are excluded from the executive findings.",
            ]
        if governance.get("review_claim_ids"):
            limitations.append(
                "Some claims remain under review and are not presented as verified facts."
            )

        return ExecutiveReportOutput(
            title=title,
            synthetic_data_notice=synthetic_notice,
            executive_summary=executive_summary,
            business_objective=business_objective,
            key_findings=report_findings,
            risks=risks,
            recommendations=recommendations,
            governance_status=(
                f"Decision: {decision}; risk: {governance.get('risk_level', 'unknown')}; "
                f"approval required: {bool(governance.get('approval_required', False))}."
            ),
            mission_success_score=score,
            mission_status=mission_status,
            next_recommended_action=next_action,
            limitations=limitations,
        )

    @staticmethod
    def _enforce_baseline(
        candidate: ExecutiveReportOutput,
        baseline: ExecutiveReportOutput,
    ) -> ExecutiveReportOutput:
        candidate.title = baseline.title
        candidate.synthetic_data_notice = baseline.synthetic_data_notice
        candidate.business_objective = baseline.business_objective
        candidate.key_findings = baseline.key_findings
        candidate.risks = baseline.risks
        candidate.recommendations = baseline.recommendations
        candidate.governance_status = baseline.governance_status
        candidate.mission_success_score = baseline.mission_success_score
        candidate.mission_status = baseline.mission_status
        candidate.next_recommended_action = baseline.next_recommended_action
        candidate.limitations = baseline.limitations
        return candidate

    def run(
        self,
        *,
        root_cause_result: dict[str, Any],
        verification_result: dict[str, Any],
        governance_result: dict[str, Any],
        evaluation_result: dict[str, Any],
        business_objective: str = (
            "Analyse customer complaints, identify recurring problems and root causes, "
            "verify findings, apply governance and prepare executive recommendations."
        ),
        workflow_profile: str = "complaint",
    ) -> AgentExecutionResult:
        started_at = perf_counter()

        baseline = self._build_baseline(
            root_cause_result=root_cause_result,
            verification_result=verification_result,
            governance_result=governance_result,
            evaluation_result=evaluation_result,
            business_objective=business_objective,
            workflow_profile=workflow_profile,
        )

        system_prompt = """
        You are the CORTEX Executive Report Agent.

        Write a concise, readable enterprise report from the supplied verified and
        governance-approved facts. Do not add facts, counts, evidence IDs, dates,
        actions or conclusions. Rejected and unapproved claims must not appear as
        verified findings. Preserve the requested structured schema. Clearly label
        the information as synthetic demonstration data.
        """

        user_prompt = (
            "Create the final executive report from this approved baseline:\n"
            + baseline.model_dump_json(indent=2)
            + "\n\nSupporting pipeline context:\n"
            + json.dumps(
                {
                    "verification": self._payload(verification_result, "verification"),
                    "governance": self._payload(governance_result, "governance"),
                    "evaluation": self._payload(evaluation_result, "evaluation"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        llm_result = self.llm_service.invoke_structured(
            agent_name=self.name,
            schema=ExecutiveReportOutput,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            mock_value=baseline,
        )

        latency_ms = int((perf_counter() - started_at) * 1000)
        if llm_result.status != "completed":
            return self.failure(
                error=f"Executive report generation failed: {llm_result.error}",
                latency_ms=latency_ms,
            )

        output = llm_result.content
        if not isinstance(output, ExecutiveReportOutput):
            try:
                output = ExecutiveReportOutput.model_validate(output)
            except Exception as exc:
                return self.failure(
                    error=f"Invalid executive report output: {exc}",
                    latency_ms=latency_ms,
                )

        output = self._enforce_baseline(output, baseline)
        evidence_ids = sorted(
            {
                evidence_id
                for finding in output.key_findings
                for evidence_id in finding.evidence_ids
            }
        )

        return self.success(
            output={
                "report": output.model_dump(),
                "provider": llm_result.provider,
                "deployment": llm_result.deployment,
                "synthetic_data": True,
            },
            confidence=output.mission_success_score / 100.0,
            latency_ms=latency_ms,
            usage=llm_result.usage,
            evidence_ids=evidence_ids,
        )
