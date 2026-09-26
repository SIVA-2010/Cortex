from __future__ import annotations

from copy import deepcopy
from time import perf_counter
from typing import Any

from backend.agents.base_agent import AgentExecutionResult, BaseAgent


class AdversarialTestAgent(BaseAgent):
    """Inject one controlled unsupported claim for hallucination-defense testing.

    This TEST ONLY agent is a fixed assurance stage in every Customer Complaint
    Intelligence workflow. It never replaces or changes the behaviour of the normal
    production agents and it consumes zero model tokens.
    """

    code = "adversarial_test_agent"
    name = "Adversarial Test Agent"

    FALSE_FINDING_ID = "RC-003"
    FALSE_EVIDENCE_ID = "TEST-FAKE-001"

    def __init__(self) -> None:
        # Deliberately do not initialise an LLM service. This is a deterministic
        # fault-injection component and therefore consumes zero model tokens.
        pass

    @staticmethod
    def _analysis_payload(root_cause_result: dict[str, Any]) -> dict[str, Any]:
        payload = root_cause_result
        if isinstance(payload.get("output"), dict):
            payload = payload["output"]
        analysis = payload.get("analysis", payload)
        if not isinstance(analysis, dict):
            raise ValueError("Root-cause analysis payload is unavailable.")
        return analysis

    @classmethod
    def _renumber_normal_findings(
        cls,
        findings: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Reserve RC-003 for the injected claim without dropping real findings."""
        prepared: list[dict[str, Any]] = []
        for index, finding in enumerate(findings, start=1):
            item = deepcopy(finding)
            item["source_agent_code"] = str(
                item.get("source_agent_code") or "root_cause_analysis"
            )

            # The adversarial claim occupies RC-003. Existing findings after the
            # first two are shifted forward only inside the adversarial test run.
            display_index = index if index <= 2 else index + 1
            item["finding_id"] = f"RC-{display_index:03d}"
            prepared.append(item)
        return prepared

    @classmethod
    def inject_unsupported_claim(
        cls,
        root_cause_result: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        modified = deepcopy(root_cause_result)
        analysis = cls._analysis_payload(modified)

        raw_findings = analysis.get("findings", [])
        if not isinstance(raw_findings, list):
            raise ValueError("Root-cause findings must be a list.")

        normal_findings = [
            item for item in raw_findings if isinstance(item, dict)
        ]
        prepared = cls._renumber_normal_findings(normal_findings)

        injected = {
            "finding_id": cls.FALSE_FINDING_ID,
            "source_agent_code": cls.code,
            "title": "Unsupported cyberattack hypothesis - controlled test",
            "category": "Payment Failure",
            "status": "confirmed",
            "root_cause": (
                "A coordinated cyberattack caused the July payment and refund failures."
            ),
            "reasoning": (
                "This statement is intentionally injected for CORTEX hallucination-defense "
                "testing. The referenced evidence ID does not exist in the approved "
                "operational evidence catalog."
            ),
            "supporting_evidence_ids": [cls.FALSE_EVIDENCE_ID],
            "complaint_evidence_ids": [],
            "confidence": 0.99,
            "business_impact": (
                "This is test-only content and must not be used for a business decision."
            ),
            "recommended_action": (
                "Reject this unsupported hypothesis unless approved evidence is supplied."
            ),
            "risk_level": "medium",
            "human_approval_required": False,
            "evidence_gaps": [
                "No approved operational evidence supports this controlled test claim."
            ],
            "test_only": True,
        }

        insert_at = min(2, len(prepared))
        prepared.insert(insert_at, injected)
        analysis["findings"] = prepared
        analysis["adversarial_test"] = {
            "enabled": True,
            "test_only": True,
            "injected_finding_id": cls.FALSE_FINDING_ID,
            "source_agent_code": cls.code,
            "submitted_evidence_id": cls.FALSE_EVIDENCE_ID,
            "expected_verification_result": "rejected",
            "purpose": "Validate claim verification and hallucination defenses.",
        }

        return modified, injected

    def run(
        self,
        *,
        root_cause_result: dict[str, Any],
    ) -> AgentExecutionResult:
        started_at = perf_counter()
        try:
            modified, injected = self.inject_unsupported_claim(root_cause_result)
        except Exception as exc:
            return self.failure(
                error=f"Adversarial test injection failed: {exc}",
                latency_ms=int((perf_counter() - started_at) * 1000),
                warnings=["Test-only adversarial stage did not modify production findings."],
            )

        return self.success(
            output={
                "root_cause_result": modified,
                "injected_finding": injected,
                "test_only": True,
                "expected_result": "rejected",
            },
            confidence=0.99,
            latency_ms=int((perf_counter() - started_at) * 1000),
            evidence_ids=[],
            warnings=[
                "TEST ONLY: continuous complaint assurance injected one intentionally unsupported claim for verification."
            ],
            estimated_cost=0.0,
        )
