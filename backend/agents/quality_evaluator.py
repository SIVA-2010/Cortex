from __future__ import annotations

from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.agents.base_agent import AgentExecutionResult, BaseAgent


MissionStatus = Literal["Excellent", "Good", "Needs Review", "Failed"]


class EvaluationOutput(BaseModel):
    mission_success_score: float = Field(ge=0.0, le=100.0)
    status: MissionStatus
    component_scores: dict[str, float]
    weakest_component: str
    recommended_improvement: str
    verified_claim_ratio: float = Field(ge=0.0, le=1.0)
    evaluation_notes: list[str] = Field(default_factory=list)


class QualityEvaluator(BaseAgent):
    code = "quality_evaluator"
    name = "Quality Evaluator"

    WEIGHTS = {
        "task_completion": 0.20,
        "output_quality_relevance": 0.20,
        "groundedness": 0.20,
        "hallucination_safety": 0.15,
        "governance_compliance": 0.10,
        "cost_efficiency": 0.10,
        "latency_efficiency": 0.05,
    }

    @staticmethod
    def _payload(value: dict[str, Any], key: str) -> dict[str, Any]:
        payload = value
        if isinstance(payload.get("output"), dict):
            payload = payload["output"]
        if isinstance(payload.get(key), dict):
            payload = payload[key]
        return payload

    @staticmethod
    def _status(score: float) -> MissionStatus:
        if score >= 90:
            return "Excellent"
        if score >= 80:
            return "Good"
        if score >= 70:
            return "Needs Review"
        return "Failed"

    @classmethod
    def calculate(
        cls,
        *,
        root_cause_result: dict[str, Any],
        verification_result: dict[str, Any],
        governance_result: dict[str, Any],
        total_tokens: int = 0,
        total_latency_ms: int = 0,
    ) -> EvaluationOutput:
        root_payload = root_cause_result
        if isinstance(root_payload.get("output"), dict):
            root_payload = root_payload["output"]
        analysis = root_payload.get("analysis", root_payload)
        findings = analysis.get("findings", []) if isinstance(analysis, dict) else []
        if not isinstance(findings, list):
            findings = []

        verification = cls._payload(verification_result, "verification")
        governance = cls._payload(governance_result, "governance")
        claims = verification.get("claims", [])
        if not isinstance(claims, list):
            claims = []

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
            for item in adversarial_claims
        )

        verified_count = sum(
            str(item.get("verification_result", "")).lower() == "verified"
            for item in operational_claims
        )
        claim_count = len(operational_claims)
        verified_ratio = verified_count / claim_count if claim_count else 0.0

        finding_confidences = [
            float(item.get("confidence", 0.0) or 0.0)
            for item in findings
            if isinstance(item, dict)
            and str(item.get("source_agent_code", "root_cause_analysis"))
            != "adversarial_test_agent"
        ]
        average_finding_confidence = (
            sum(finding_confidences) / len(finding_confidences)
            if finding_confidences
            else 0.0
        )

        groundedness_values = [
            float(item.get("groundedness", 0.0) or 0.0)
            for item in operational_claims
        ]
        hallucination_values = [
            float(item.get("hallucination_risk", 1.0) or 1.0)
            for item in operational_claims
        ]
        groundedness = (
            sum(groundedness_values) / len(groundedness_values)
            if groundedness_values
            else 0.0
        )
        hallucination_risk = (
            sum(hallucination_values) / len(hallucination_values)
            if hallucination_values
            else 1.0
        )
        governance_decision = str(governance.get("decision", "block"))

        completed_stages = 4
        if findings:
            completed_stages += 1
        if claims:
            completed_stages += 1
        if governance:
            completed_stages += 1
        task_completion = min(100.0, completed_stages / 7 * 100.0)

        output_quality = min(
            100.0,
            average_finding_confidence * 70.0 + verified_ratio * 30.0,
        )
        groundedness_score = groundedness * 100.0
        hallucination_safety = max(0.0, (1.0 - hallucination_risk) * 100.0)

        governance_compliance = {
            "allow": 100.0,
            "approval_required": 100.0,
            "verify": 82.0,
            "block": 20.0,
        }.get(governance_decision, 20.0)

        if total_tokens <= 0 or total_tokens <= 4000:
            cost_efficiency = 100.0
        elif total_tokens <= 8000:
            cost_efficiency = 90.0
        elif total_tokens <= 15000:
            cost_efficiency = 80.0
        else:
            cost_efficiency = 65.0

        if total_latency_ms <= 0 or total_latency_ms <= 15000:
            latency_efficiency = 100.0
        elif total_latency_ms <= 30000:
            latency_efficiency = 90.0
        elif total_latency_ms <= 60000:
            latency_efficiency = 80.0
        else:
            latency_efficiency = 65.0

        component_scores = {
            "task_completion": round(task_completion, 2),
            "output_quality_relevance": round(output_quality, 2),
            "groundedness": round(groundedness_score, 2),
            "hallucination_safety": round(hallucination_safety, 2),
            "governance_compliance": round(governance_compliance, 2),
            "cost_efficiency": round(cost_efficiency, 2),
            "latency_efficiency": round(latency_efficiency, 2),
        }

        score = sum(
            component_scores[name] * weight
            for name, weight in cls.WEIGHTS.items()
        )
        weakest = min(component_scores, key=component_scores.get)

        recommendations = {
            "task_completion": "Complete every planned workflow stage before reporting.",
            "output_quality_relevance": "Improve finding precision and remove low-confidence conclusions.",
            "groundedness": "Attach more direct and independent operational evidence.",
            "hallucination_safety": "Reject unsupported claims and rerun verification.",
            "governance_compliance": "Resolve policy failures or obtain required approval.",
            "cost_efficiency": "Reduce prompt size and reuse deterministic analytics.",
            "latency_efficiency": "Parallelise independent retrieval and analysis nodes.",
        }

        return EvaluationOutput(
            mission_success_score=round(score, 2),
            status=cls._status(score),
            component_scores=component_scores,
            weakest_component=weakest,
            recommended_improvement=recommendations[weakest],
            verified_claim_ratio=round(verified_ratio, 4),
            evaluation_notes=[
                "Mission score uses the CORTEX weighted evaluation framework.",
                f"Verified production-claim ratio: {verified_ratio:.2%}.",
                (
                    "Continuous adversarial assurance passed: the TEST ONLY claim was rejected and excluded from production quality scoring."
                    if adversarial_passed
                    else "No completed adversarial assurance result was available for this execution."
                ),
                f"Governance decision: {governance_decision}.",
            ],
        )

    def run(
        self,
        *,
        root_cause_result: dict[str, Any],
        verification_result: dict[str, Any],
        governance_result: dict[str, Any],
        total_tokens: int = 0,
        total_latency_ms: int = 0,
    ) -> AgentExecutionResult:
        started_at = perf_counter()

        try:
            output = self.calculate(
                root_cause_result=root_cause_result,
                verification_result=verification_result,
                governance_result=governance_result,
                total_tokens=total_tokens,
                total_latency_ms=total_latency_ms,
            )
        except Exception as exc:
            return self.failure(
                error=f"Mission evaluation failed: {exc}",
                latency_ms=int((perf_counter() - started_at) * 1000),
            )

        latency_ms = int((perf_counter() - started_at) * 1000)
        return self.success(
            output={
                "evaluation": output.model_dump(),
                "synthetic_data": True,
            },
            confidence=output.mission_success_score / 100.0,
            latency_ms=latency_ms,
        )
