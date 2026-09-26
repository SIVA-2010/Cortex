from __future__ import annotations

import json
from statistics import mean
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.agents.base_agent import AgentExecutionResult, BaseAgent


CitationStatus = Literal["supported", "partial", "missing", "contradicted"]
VerificationStatus = Literal["verified", "needs_review", "rejected"]
RootCauseStatus = Literal["confirmed", "probable", "unverified"]


class ClaimVerification(BaseModel):
    claim_id: str
    finding_id: str
    category: str
    claim: str
    source_agent_code: str = "root_cause_analysis"
    root_cause_status: RootCauseStatus
    evidence_ids: list[str] = Field(default_factory=list)
    invalid_evidence_ids: list[str] = Field(default_factory=list)
    complaint_evidence_ids: list[str] = Field(default_factory=list)
    evidence_types: list[str] = Field(default_factory=list)
    direct_evidence_count: int = Field(default=0, ge=0)
    citation_status: CitationStatus
    groundedness: float = Field(ge=0.0, le=1.0)
    contradiction_found: bool = False
    hallucination_risk: float = Field(ge=0.0, le=1.0)
    verification_result: VerificationStatus
    explanation: str
    evidence_gaps: list[str] = Field(default_factory=list)


class VerificationOutput(BaseModel):
    executive_summary: str
    claims: list[ClaimVerification]
    verified_claim_count: int = Field(ge=0)
    review_required_count: int = Field(ge=0)
    rejected_claim_count: int = Field(ge=0)
    overall_groundedness: float = Field(ge=0.0, le=1.0)
    overall_hallucination_risk: float = Field(ge=0.0, le=1.0)


class VerificationAgent(BaseAgent):
    code = "verification"
    name = "Verification Agent"

    @staticmethod
    def _extract_sections(
        root_cause_result: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        payload = root_cause_result

        if isinstance(payload.get("output"), dict):
            payload = payload["output"]

        analysis = payload.get("analysis", payload)
        if not isinstance(analysis, dict):
            analysis = {}

        findings = analysis.get("findings", [])
        if not isinstance(findings, list):
            raise ValueError("Root-cause findings must be a list.")

        operational = (
            payload.get("operational_evidence_summary")
            or payload.get("operational_summary")
            or payload.get("operational_evidence")
            or {}
        )
        if not isinstance(operational, dict):
            operational = {}

        catalog = operational.get("evidence_catalog", [])
        if not isinstance(catalog, list) or not catalog:
            catalog = []
            for key, evidence_type in (
                ("incident_evidence", "service_incident"),
                ("release_evidence", "product_release"),
                ("transaction_signals", "transaction_signal"),
                ("support_signals", "support_signal"),
            ):
                values = operational.get(key, [])
                if not isinstance(values, list):
                    continue
                for value in values:
                    if isinstance(value, dict):
                        item = dict(value)
                        item.setdefault("evidence_type", evidence_type)
                        catalog.append(item)

        evidence_by_id: dict[str, dict[str, Any]] = {}
        for item in catalog:
            if not isinstance(item, dict):
                continue
            evidence_id = str(item.get("evidence_id", "")).strip()
            if evidence_id:
                evidence_by_id[evidence_id] = item

        return findings, evidence_by_id

    @staticmethod
    def _normalize_status(value: Any) -> RootCauseStatus:
        status = str(value or "unverified").strip().lower()
        if status not in {"confirmed", "probable", "unverified"}:
            return "unverified"
        return status  # type: ignore[return-value]

    @staticmethod
    def _finding_value(
        finding: dict[str, Any],
        *names: str,
        default: Any = None,
    ) -> Any:
        for name in names:
            value = finding.get(name)
            if value is not None:
                return value
        return default

    @staticmethod
    def _has_contradiction(evidence: list[dict[str, Any]]) -> bool:
        contradiction_phrases = (
            "false alarm",
            "false positive",
            "unrelated",
            "no causal link",
            "cause rejected",
            "not associated",
        )

        for item in evidence:
            text = " ".join(str(value) for value in item.values()).lower()
            if any(phrase in text for phrase in contradiction_phrases):
                return True
        return False

    @classmethod
    def _verify_finding(
        cls,
        finding: dict[str, Any],
        evidence_by_id: dict[str, dict[str, Any]],
        index: int,
    ) -> ClaimVerification:
        finding_id = str(
            cls._finding_value(
                finding,
                "finding_id",
                "id",
                default=f"RC-{index:03d}",
            )
        )
        claim_id = f"CLM-{index:03d}"
        category = str(
            cls._finding_value(
                finding,
                "category",
                "complaint_category",
                default="Uncategorised",
            )
        )
        claim = str(
            cls._finding_value(
                finding,
                "root_cause",
                "claim",
                default="",
            )
        ).strip()
        source_agent_code = str(
            cls._finding_value(
                finding,
                "source_agent_code",
                default="root_cause_analysis",
            )
        ).strip() or "root_cause_analysis"
        root_status = cls._normalize_status(finding.get("status"))

        supplied_ids = cls._finding_value(
            finding,
            "supporting_evidence_ids",
            "evidence_ids",
            default=[],
        )
        if not isinstance(supplied_ids, list):
            supplied_ids = []
        supplied_ids = [str(value).strip() for value in supplied_ids if str(value).strip()]

        complaint_ids = finding.get("complaint_evidence_ids", [])
        if not isinstance(complaint_ids, list):
            complaint_ids = []
        complaint_ids = [str(value).strip() for value in complaint_ids if str(value).strip()]

        valid_ids = [value for value in supplied_ids if value in evidence_by_id]
        invalid_ids = sorted(set(supplied_ids) - set(valid_ids))
        valid_evidence = [evidence_by_id[value] for value in valid_ids]

        evidence_types = sorted(
            {
                str(item.get("evidence_type", "unknown"))
                for item in valid_evidence
            }
        )
        direct_count = sum(
            str(item.get("strength", "")).lower() == "direct"
            for item in valid_evidence
        )
        contradiction_found = cls._has_contradiction(valid_evidence)

        if not claim:
            groundedness = 0.0
            citation_status: CitationStatus = "missing"
            verification_result: VerificationStatus = "rejected"
            explanation = "The finding did not contain a root-cause claim."
        elif not valid_ids:
            groundedness = 0.05 if root_status != "unverified" else 0.15
            citation_status = "missing"
            verification_result = (
                "rejected" if root_status in {"confirmed", "probable"} else "needs_review"
            )
            explanation = "No valid operational evidence ID supports the claim."
        else:
            groundedness = 0.25
            groundedness += min(len(valid_ids) * 0.11, 0.33)
            groundedness += min(len(evidence_types) * 0.08, 0.24)
            groundedness += min(direct_count * 0.12, 0.24)

            if root_status == "confirmed":
                groundedness += 0.04
            elif root_status == "unverified":
                groundedness -= 0.12

            if invalid_ids:
                groundedness -= 0.10
            if contradiction_found:
                groundedness -= 0.35

            groundedness = max(0.0, min(0.98, groundedness))

            if contradiction_found:
                citation_status = "contradicted"
                verification_result = "rejected"
                explanation = "Available evidence contains a contradiction signal."
            else:
                citation_status = "partial" if invalid_ids else "supported"

                if (
                    root_status == "confirmed"
                    and direct_count >= 1
                    and len(valid_ids) >= 2
                    and groundedness >= 0.72
                ):
                    verification_result = "verified"
                elif (
                    root_status == "probable"
                    and len(valid_ids) >= 2
                    and len(evidence_types) >= 2
                    and groundedness >= 0.60
                ):
                    verification_result = "verified"
                elif root_status == "unverified":
                    verification_result = "needs_review"
                else:
                    verification_result = "needs_review"

                explanation = (
                    f"The claim references {len(valid_ids)} valid evidence item(s) "
                    f"across {len(evidence_types)} evidence type(s), including "
                    f"{direct_count} direct evidence item(s)."
                )

        evidence_gaps: list[str] = []
        if invalid_ids:
            evidence_gaps.append(
                "Unsupported evidence IDs were removed: " + ", ".join(invalid_ids)
            )
        if root_status == "confirmed" and direct_count == 0:
            evidence_gaps.append("A confirmed classification requires direct evidence.")
        if len(evidence_types) < 2 and root_status == "probable":
            evidence_gaps.append("A probable claim needs independent corroboration.")
        if not complaint_ids:
            evidence_gaps.append("No representative complaint IDs were attached.")

        hallucination_risk = max(0.0, min(1.0, 1.0 - groundedness))
        if contradiction_found:
            hallucination_risk = max(hallucination_risk, 0.85)

        return ClaimVerification(
            claim_id=claim_id,
            finding_id=finding_id,
            category=category,
            claim=claim or "No root-cause claim supplied.",
            source_agent_code=source_agent_code,
            root_cause_status=root_status,
            evidence_ids=valid_ids,
            invalid_evidence_ids=invalid_ids,
            complaint_evidence_ids=complaint_ids,
            evidence_types=evidence_types,
            direct_evidence_count=direct_count,
            citation_status=citation_status,
            groundedness=round(groundedness, 4),
            contradiction_found=contradiction_found,
            hallucination_risk=round(hallucination_risk, 4),
            verification_result=verification_result,
            explanation=explanation,
            evidence_gaps=evidence_gaps,
        )

    @classmethod
    def _build_deterministic_output(
        cls,
        findings: list[dict[str, Any]],
        evidence_by_id: dict[str, dict[str, Any]],
    ) -> VerificationOutput:
        claims = [
            cls._verify_finding(finding, evidence_by_id, index)
            for index, finding in enumerate(findings, start=1)
            if isinstance(finding, dict)
        ]

        verified = sum(item.verification_result == "verified" for item in claims)
        review = sum(item.verification_result == "needs_review" for item in claims)
        rejected = sum(item.verification_result == "rejected" for item in claims)
        groundedness = mean(item.groundedness for item in claims) if claims else 0.0
        hallucination = (
            mean(item.hallucination_risk for item in claims) if claims else 1.0
        )

        return VerificationOutput(
            executive_summary=(
                f"CORTEX verified {verified} claim(s), routed {review} claim(s) "
                f"for review and rejected {rejected} unsupported claim(s)."
            ),
            claims=claims,
            verified_claim_count=verified,
            review_required_count=review,
            rejected_claim_count=rejected,
            overall_groundedness=round(groundedness, 4),
            overall_hallucination_risk=round(hallucination, 4),
        )

    @staticmethod
    def _merge_llm_narrative(
        candidate: VerificationOutput,
        baseline: VerificationOutput,
    ) -> VerificationOutput:
        candidate_by_id = {item.claim_id: item for item in candidate.claims}
        final_claims: list[ClaimVerification] = []

        for baseline_claim in baseline.claims:
            candidate_claim = candidate_by_id.get(baseline_claim.claim_id)
            if candidate_claim is None:
                final_claims.append(baseline_claim)
                continue

            candidate_claim.finding_id = baseline_claim.finding_id
            candidate_claim.category = baseline_claim.category
            candidate_claim.claim = baseline_claim.claim
            candidate_claim.source_agent_code = baseline_claim.source_agent_code
            candidate_claim.root_cause_status = baseline_claim.root_cause_status
            candidate_claim.evidence_ids = baseline_claim.evidence_ids
            candidate_claim.invalid_evidence_ids = baseline_claim.invalid_evidence_ids
            candidate_claim.complaint_evidence_ids = baseline_claim.complaint_evidence_ids
            candidate_claim.evidence_types = baseline_claim.evidence_types
            candidate_claim.direct_evidence_count = baseline_claim.direct_evidence_count
            candidate_claim.citation_status = baseline_claim.citation_status
            candidate_claim.groundedness = baseline_claim.groundedness
            candidate_claim.contradiction_found = baseline_claim.contradiction_found
            candidate_claim.hallucination_risk = baseline_claim.hallucination_risk
            candidate_claim.verification_result = baseline_claim.verification_result
            candidate_claim.evidence_gaps = baseline_claim.evidence_gaps

            if not candidate_claim.explanation.strip():
                candidate_claim.explanation = baseline_claim.explanation

            final_claims.append(candidate_claim)

        candidate.claims = final_claims
        candidate.verified_claim_count = baseline.verified_claim_count
        candidate.review_required_count = baseline.review_required_count
        candidate.rejected_claim_count = baseline.rejected_claim_count
        candidate.overall_groundedness = baseline.overall_groundedness
        candidate.overall_hallucination_risk = baseline.overall_hallucination_risk
        return candidate

    def run(
        self,
        *,
        root_cause_result: dict[str, Any],
    ) -> AgentExecutionResult:
        started_at = perf_counter()

        try:
            findings, evidence_by_id = self._extract_sections(root_cause_result)
        except Exception as exc:
            return self.failure(
                error=f"Verification input processing failed: {exc}",
                latency_ms=int((perf_counter() - started_at) * 1000),
            )

        baseline = self._build_deterministic_output(findings, evidence_by_id)

        system_prompt = """
        You are the CORTEX Verification Agent.

        Review each root-cause claim against the supplied evidence catalog.
        Use only the supplied evidence. Do not invent evidence IDs, complaint IDs,
        counts, dates, incidents or releases. Explain evidence coverage and gaps in
        concise enterprise language. Preserve the requested structured schema.

        Confirmed claims require direct operational evidence. Probable claims need
        independent corroboration. Unsupported or contradicted claims must be marked
        for review or rejected. All information is synthetic demonstration data.
        """

        user_prompt = (
            "Root-cause findings:\n"
            + json.dumps(findings, ensure_ascii=False, indent=2)
            + "\n\nApproved operational evidence catalog:\n"
            + json.dumps(list(evidence_by_id.values()), ensure_ascii=False, indent=2)
            + "\n\nDeterministic verification baseline:\n"
            + baseline.model_dump_json(indent=2)
        )

        llm_result = self.llm_service.invoke_structured(
            agent_name=self.name,
            schema=VerificationOutput,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            mock_value=baseline,
        )

        latency_ms = int((perf_counter() - started_at) * 1000)

        if llm_result.status != "completed":
            return self.failure(
                error=f"Claim verification failed: {llm_result.error}",
                latency_ms=latency_ms,
            )

        output = llm_result.content
        if not isinstance(output, VerificationOutput):
            try:
                output = VerificationOutput.model_validate(output)
            except Exception as exc:
                return self.failure(
                    error=f"Invalid verification output: {exc}",
                    latency_ms=latency_ms,
                )

        output = self._merge_llm_narrative(output, baseline)
        evidence_ids = sorted(
            {
                evidence_id
                for claim in output.claims
                for evidence_id in claim.evidence_ids
            }
        )

        return self.success(
            output={
                "verification": output.model_dump(),
                "provider": llm_result.provider,
                "deployment": llm_result.deployment,
                "synthetic_data": True,
            },
            confidence=output.overall_groundedness,
            latency_ms=latency_ms,
            usage=llm_result.usage,
            evidence_ids=evidence_ids,
        )
