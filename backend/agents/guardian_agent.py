from __future__ import annotations

from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.agents.base_agent import AgentExecutionResult, BaseAgent


PolicyStatus = Literal["pass", "warning", "fail"]
Decision = Literal["allow", "verify", "approval_required", "block"]
RiskLevel = Literal["low", "medium", "high", "prohibited"]


class PolicyCheck(BaseModel):
    policy: str
    status: PolicyStatus
    detail: str


class GovernanceOutput(BaseModel):
    summary: str
    decision: Decision
    risk_level: RiskLevel
    approval_required: bool
    approval_reason: str | None = None
    approval_payload: dict[str, Any] | None = None
    policy_checks: list[PolicyCheck]
    permitted_claim_ids: list[str] = Field(default_factory=list)
    review_claim_ids: list[str] = Field(default_factory=list)
    excluded_claim_ids: list[str] = Field(default_factory=list)
    blocked_recommendations: list[str] = Field(default_factory=list)
    pii_status: str


class GuardianGovernanceAgent(BaseAgent):
    code = "guardian_governance"
    name = "Guardian Governance Agent"

    HIGH_IMPACT_TERMS = (
        "rollback",
        "shut down",
        "shutdown",
        "disable service",
        "revoke access",
        "access revocation",
        "production change",
        "deploy patch",
        "financial transaction",
    )

    @staticmethod
    def _payload(value: dict[str, Any], nested_key: str) -> dict[str, Any]:
        payload = value
        if isinstance(payload.get("output"), dict):
            payload = payload["output"]
        if isinstance(payload.get(nested_key), dict):
            payload = payload[nested_key]
        return payload

    @classmethod
    def run_policy_engine(
        cls,
        *,
        root_cause_result: dict[str, Any],
        verification_result: dict[str, Any],
        complaint_output: dict[str, Any] | None = None,
        workflow_profile: str = "complaint",
    ) -> GovernanceOutput:
        root_payload = root_cause_result
        if isinstance(root_payload.get("output"), dict):
            root_payload = root_payload["output"]
        analysis = root_payload.get("analysis", root_payload)
        findings = analysis.get("findings", []) if isinstance(analysis, dict) else []
        if not isinstance(findings, list):
            findings = []

        verification = cls._payload(verification_result, "verification")
        claims = verification.get("claims", [])
        if not isinstance(claims, list):
            claims = []

        profile = str(workflow_profile or "complaint").lower()
        if profile == "logistics":
            rows_with_pii = 0
            masking_applied = True
            pii_failure = False
            pii_status = "not_applicable"
        else:
            complaint_payload = complaint_output or {}
            if isinstance(complaint_payload.get("output"), dict):
                complaint_payload = complaint_payload["output"]
            dataset_summary = complaint_payload.get("dataset_summary", {})
            if not isinstance(dataset_summary, dict):
                dataset_summary = {}
            pii_summary = dataset_summary.get("pii_summary", {})
            if not isinstance(pii_summary, dict):
                pii_summary = {}

            rows_with_pii = int(pii_summary.get("rows_with_detected_pii", 0) or 0)
            masking_applied = bool(pii_summary.get("masking_applied", rows_with_pii == 0))
            pii_failure = rows_with_pii > 0 and not masking_applied
            pii_status = "failed" if pii_failure else "masked"

        permitted_claim_ids = [
            str(item.get("claim_id"))
            for item in claims
            if isinstance(item, dict)
            and item.get("verification_result") == "verified"
        ]
        review_claim_ids = [
            str(item.get("claim_id"))
            for item in claims
            if isinstance(item, dict)
            and item.get("verification_result") == "needs_review"
        ]
        excluded_claim_ids = [
            str(item.get("claim_id"))
            for item in claims
            if isinstance(item, dict)
            and item.get("verification_result") == "rejected"
        ]

        # The adversarial claim is deliberately hostile test input. It must be
        # excluded and audited, but a successful rejection must not make the
        # production complaint findings look unsafe. Global governance thresholds
        # therefore use only non-test claims.
        operational_claims = [
            item
            for item in claims
            if isinstance(item, dict)
            and str(item.get("source_agent_code", "")) != "adversarial_test_agent"
        ]
        adversarial_claims = [
            item
            for item in claims
            if isinstance(item, dict)
            and str(item.get("source_agent_code", "")) == "adversarial_test_agent"
        ]
        adversarial_passed = bool(adversarial_claims) and all(
            str(item.get("verification_result", "")).lower() == "rejected"
            and bool(item.get("invalid_evidence_ids"))
            for item in adversarial_claims
        )

        max_hallucination = max(
            [
                float(item.get("hallucination_risk", 0.0) or 0.0)
                for item in operational_claims
            ]
            or [0.0]
        )
        operational_groundedness = [
            float(item.get("groundedness", 0.0) or 0.0)
            for item in operational_claims
        ]
        overall_groundedness = (
            sum(operational_groundedness) / len(operational_groundedness)
            if operational_groundedness
            else 0.0
        )

        high_impact_actions: list[str] = []
        blocked_recommendations: list[str] = []
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            action = str(finding.get("recommended_action", "")).strip()
            action_lower = action.lower()
            risk = str(finding.get("risk_level", "")).lower()
            approval_level = str(finding.get("approval_level", "")).lower()
            explicit_approval = bool(finding.get("human_approval_required", False))

            is_high_impact = (
                risk in {"high", "critical"}
                or approval_level in {"reviewer", "administrator", "admin"}
                or explicit_approval
                or any(term in action_lower for term in cls.HIGH_IMPACT_TERMS)
            )
            if is_high_impact and action:
                high_impact_actions.append(action)

        policy_checks = [
            PolicyCheck(
                policy="PII masking",
                status="fail" if pii_failure else "pass",
                detail=(
                    "Detected PII was not masked before AI processing."
                    if pii_failure
                    else (
                        "Complaint PII masking control is not applicable to the synthetic logistics workflow."
                        if profile == "logistics"
                        else "PII masking controls passed for the supplied complaint data."
                    )
                ),
            ),
            PolicyCheck(
                policy="Claim verification",
                status=(
                    "fail"
                    if not permitted_claim_ids and excluded_claim_ids
                    else "warning"
                    if review_claim_ids
                    else "pass"
                ),
                detail=(
                    f"{len(permitted_claim_ids)} verified, "
                    f"{len(review_claim_ids)} require review and "
                    f"{len(excluded_claim_ids)} were excluded."
                ),
            ),
            PolicyCheck(
                policy="Hallucination threshold",
                status=(
                    "fail"
                    if max_hallucination > 0.75
                    else "warning"
                    if max_hallucination > 0.45
                    else "pass"
                ),
                detail=f"Maximum claim hallucination risk is {max_hallucination:.2f}.",
            ),
            PolicyCheck(
                policy="Groundedness threshold",
                status="warning" if overall_groundedness < 0.65 else "pass",
                detail=f"Overall groundedness is {overall_groundedness:.2f}.",
            ),
            PolicyCheck(
                policy="Continuous adversarial assurance",
                status=(
                    "pass"
                    if adversarial_passed
                    else "warning"
                    if adversarial_claims
                    else "pass"
                ),
                detail=(
                    "TEST ONLY adversarial RC-003 was rejected with an invalid evidence reference; production claims remain independently governed."
                    if adversarial_passed
                    else "No adversarial challenge was present in this legacy/non-complaint execution."
                    if not adversarial_claims
                    else "The adversarial challenge did not reach the expected rejected state."
                ),
            ),
            PolicyCheck(
                policy="High-impact action control",
                status="warning" if high_impact_actions else "pass",
                detail=(
                    f"{len(high_impact_actions)} action(s) require human approval."
                    if high_impact_actions
                    else "No high-impact action requires approval."
                ),
            ),
        ]

        if pii_failure:
            decision: Decision = "block"
            risk_level: RiskLevel = "prohibited"
            approval_required = False
            approval_reason = "Unmasked personal data cannot proceed."
        elif not permitted_claim_ids and excluded_claim_ids:
            decision = "block"
            risk_level = "prohibited"
            approval_required = False
            approval_reason = "No verified claim is available for decision support."
        elif high_impact_actions:
            decision = "approval_required"
            risk_level = "high"
            approval_required = True
            approval_reason = (
                "At least one recommendation could alter a production service, "
                "customer access or financial outcome."
            )
        elif review_claim_ids or max_hallucination > 0.45 or overall_groundedness < 0.65:
            decision = "verify"
            risk_level = "medium"
            approval_required = False
            approval_reason = "Additional verification is required before use."
        else:
            decision = "allow"
            risk_level = "low"
            approval_required = False
            approval_reason = None

        if decision == "block":
            blocked_recommendations = [
                str(item.get("recommended_action", ""))
                for item in findings
                if isinstance(item, dict) and item.get("recommended_action")
            ]

        approval_payload = None
        if approval_required:
            approval_payload = {
                "approval_type": "high_impact_recommendation",
                "reason": approval_reason,
                "actions": high_impact_actions,
                "required_role": "reviewer",
                "status": "pending",
            }

        return GovernanceOutput(
            summary=(
                f"Governance decision: {decision}. Risk level: {risk_level}. "
                f"{len(permitted_claim_ids)} claim(s) may be used in the final report."
            ),
            decision=decision,
            risk_level=risk_level,
            approval_required=approval_required,
            approval_reason=approval_reason,
            approval_payload=approval_payload,
            policy_checks=policy_checks,
            permitted_claim_ids=permitted_claim_ids,
            review_claim_ids=review_claim_ids,
            excluded_claim_ids=excluded_claim_ids,
            blocked_recommendations=blocked_recommendations,
            pii_status=pii_status,
        )

    def run(
        self,
        *,
        root_cause_result: dict[str, Any],
        verification_result: dict[str, Any],
        complaint_output: dict[str, Any] | None = None,
        workflow_profile: str = "complaint",
    ) -> AgentExecutionResult:
        started_at = perf_counter()

        try:
            output = self.run_policy_engine(
                root_cause_result=root_cause_result,
                verification_result=verification_result,
                complaint_output=complaint_output,
                workflow_profile=workflow_profile,
            )
        except Exception as exc:
            return self.failure(
                error=f"Governance evaluation failed: {exc}",
                latency_ms=int((perf_counter() - started_at) * 1000),
            )

        latency_ms = int((perf_counter() - started_at) * 1000)
        confidence = 1.0 if output.decision in {"allow", "approval_required"} else 0.85

        return self.success(
            output={
                "governance": output.model_dump(),
                "synthetic_data": True,
            },
            confidence=confidence,
            latency_ms=latency_ms,
            warnings=(
                ["Human approval is required before a high-impact action is used."]
                if output.approval_required
                else []
            ),
        )
