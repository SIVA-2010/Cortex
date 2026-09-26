from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, Field

from backend.agents.base_agent import AgentExecutionResult, BaseAgent
from backend.services.complaint_processor import ComplaintProcessor


SeverityLevel = Literal["low", "medium", "high", "critical"]


class ComplaintTaxonomyItem(BaseModel):
    category: str
    description: str
    volume: int = Field(ge=0)
    percentage: float = Field(ge=0.0, le=100.0)
    severity: SeverityLevel


class RecurringIssue(BaseModel):
    issue: str
    category: str
    volume: int = Field(ge=0)
    business_impact: str
    evidence_complaint_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class EmergingIssue(BaseModel):
    issue: str
    category: str
    recent_7_day_count: int = Field(ge=0)
    previous_7_day_count: int = Field(ge=0)
    growth_percent: float
    risk: SeverityLevel
    evidence_complaint_ids: list[str] = Field(default_factory=list)


class ComplaintIntelligenceOutput(BaseModel):
    executive_summary: str
    taxonomy: list[ComplaintTaxonomyItem]
    recurring_issues: list[RecurringIssue]
    emerging_issues: list[EmergingIssue]
    sla_findings: list[str]
    customer_impact: list[str]
    data_quality_notes: list[str]
    next_analysis_questions: list[str]


class ComplaintIntelligenceAgent(BaseAgent):
    code = "complaint_intelligence"
    name = "Complaint Intelligence Agent"

    def __init__(
        self,
        processor: ComplaintProcessor | None = None,
    ) -> None:
        super().__init__()
        self.processor = processor or ComplaintProcessor()

    @staticmethod
    def _severity_from_metric(metric: dict) -> SeverityLevel:
        critical_count = int(metric.get("critical_count", 0))
        breach_rate = float(metric.get("sla_breach_rate", 0.0))
        escalation_rate = float(metric.get("escalation_rate", 0.0))

        if critical_count > 0 or escalation_rate >= 60:
            return "critical"
        if breach_rate >= 45 or escalation_rate >= 35:
            return "high"
        if breach_rate >= 20:
            return "medium"
        return "low"

    @classmethod
    def _build_mock_output(
        cls,
        summary: dict,
    ) -> ComplaintIntelligenceOutput:
        category_metrics = summary.get("category_metrics", [])
        sample_rows = summary.get("representative_samples", [])
        samples_by_category: dict[str, list[str]] = {}

        for sample in sample_rows:
            category = str(sample.get("category", "Other"))
            samples_by_category.setdefault(category, []).append(
                str(sample.get("complaint_id", ""))
            )

        taxonomy: list[ComplaintTaxonomyItem] = []
        recurring: list[RecurringIssue] = []

        for metric in category_metrics:
            category = str(metric["category"])
            severity = cls._severity_from_metric(metric)
            taxonomy.append(
                ComplaintTaxonomyItem(
                    category=category,
                    description=(
                        f"Complaints grouped under {category.lower()} "
                        "using deterministic keyword and operational signals."
                    ),
                    volume=int(metric["count"]),
                    percentage=float(metric["percentage"]),
                    severity=severity,
                )
            )

        for metric in category_metrics[:3]:
            category = str(metric["category"])
            recurring.append(
                RecurringIssue(
                    issue=f"Recurring {category.lower()} complaints",
                    category=category,
                    volume=int(metric["count"]),
                    business_impact=(
                        f"SLA breach rate is {metric['sla_breach_rate']}% and "
                        f"the escalation rate is {metric['escalation_rate']}%."
                    ),
                    evidence_complaint_ids=samples_by_category.get(
                        category,
                        [],
                    )[:5],
                    confidence=0.90,
                )
            )

        emerging = [
            EmergingIssue(
                issue=f"Rapid increase in {item['category'].lower()}",
                category=str(item["category"]),
                recent_7_day_count=int(item["recent_7_day_count"]),
                previous_7_day_count=int(item["previous_7_day_count"]),
                growth_percent=float(item["growth_percent"]),
                risk=(
                    "high"
                    if float(item.get("sla_breach_rate", 0.0)) >= 40
                    else "medium"
                ),
                evidence_complaint_ids=list(
                    item.get("sample_complaint_ids", [])
                ),
            )
            for item in summary.get("emerging_issues", [])[:3]
        ]

        overall = summary.get("overall_metrics", {})
        highest_volume = (
            summary.get("decision_signals", {})
            .get("highest_volume_category", {})
            .get("category", "the leading category")
        )

        return ComplaintIntelligenceOutput(
            executive_summary=(
                f"The dataset contains {summary.get('valid_row_count', 0)} "
                f"validated complaints. {highest_volume} has the highest "
                "volume. The analysis identifies recurring patterns, SLA "
                "risk and emerging issues without asserting root causes."
            ),
            taxonomy=taxonomy,
            recurring_issues=recurring,
            emerging_issues=emerging,
            sla_findings=[
                f"{overall.get('sla_breach_count', 0)} complaints breached "
                f"SLA, representing {overall.get('sla_breach_rate', 0)}% "
                "of validated records.",
                f"Average first response time was "
                f"{overall.get('average_first_response_minutes', 0)} minutes.",
            ],
            customer_impact=[
                f"{overall.get('escalated_count', 0)} complaints were "
                "escalated or met escalation conditions.",
                f"Total recorded refund exposure was "
                f"{overall.get('total_refund_amount', 0)}.",
                f"Average repeat contacts were "
                f"{overall.get('average_repeat_contacts', 0)} per complaint.",
            ],
            data_quality_notes=list(
                summary.get("data_quality", {}).get("warnings", [])
            )
            or ["No material data-quality warning was detected."],
            next_analysis_questions=[
                "Which operational incidents overlap with the complaint spikes?",
                "Which product releases occurred before the leading patterns?",
                "Which findings can be confirmed with direct system evidence?",
            ],
        )

    def run(
        self,
        *,
        csv_path: str | Path,
    ) -> AgentExecutionResult:
        started_at = perf_counter()

        try:
            processed = self.processor.process_csv(csv_path)
        except Exception as exc:
            return self.failure(
                error=f"Complaint processing failed: {exc}",
                latency_ms=int((perf_counter() - started_at) * 1000),
            )

        summary = processed.summary
        mock_output = self._build_mock_output(summary)

        system_prompt = """
        You are the CORTEX Complaint Intelligence Agent.

        Analyse only the supplied deterministic dataset summary and masked
        representative complaint samples.

        Your responsibilities are:
        1. Produce a practical complaint taxonomy.
        2. Identify recurring complaint patterns.
        3. Identify fast-growing or emerging issues.
        4. Explain SLA and customer-impact signals.
        5. Cite complaint IDs as evidence where available.
        6. Separate observed patterns from possible root causes.

        Rules:
        - Do not claim a root cause; a separate Root-Cause Agent handles that.
        - Do not invent complaint counts, percentages or IDs.
        - Treat all data as synthetic demonstration data.
        - Do not reproduce personal information.
        - Use concise enterprise language.
        """

        user_prompt = (
            "Analyse this synthetic complaint dataset summary and return the "
            "required structured intelligence output.\n\n"
            + json.dumps(summary, ensure_ascii=False, indent=2)
        )

        llm_result = self.llm_service.invoke_structured(
            agent_name=self.name,
            schema=ComplaintIntelligenceOutput,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            mock_value=mock_output,
        )

        total_latency_ms = int((perf_counter() - started_at) * 1000)

        if llm_result.status != "completed":
            return self.failure(
                error=(
                    "Complaint intelligence generation failed: "
                    f"{llm_result.error}"
                ),
                latency_ms=total_latency_ms,
                warnings=list(
                    summary.get("data_quality", {}).get("warnings", [])
                ),
            )

        output = llm_result.content
        if not isinstance(output, ComplaintIntelligenceOutput):
            try:
                output = ComplaintIntelligenceOutput.model_validate(output)
            except Exception as exc:
                return self.failure(
                    error=f"Invalid complaint intelligence output: {exc}",
                    latency_ms=total_latency_ms,
                )

        valid_rows = int(summary.get("valid_row_count", 0))
        other_count = next(
            (
                int(item["count"])
                for item in summary.get("category_metrics", [])
                if item["category"] == "Other"
            ),
            0,
        )
        classified_rate = 1.0 - (other_count / max(valid_rows, 1))
        size_factor = min(valid_rows / 1000, 1.0)
        confidence = min(
            0.72 + (classified_rate * 0.18) + (size_factor * 0.08),
            0.98,
        )

        evidence_ids = sorted(
            {
                complaint_id
                for item in output.recurring_issues
                for complaint_id in item.evidence_complaint_ids
            }
            | {
                complaint_id
                for item in output.emerging_issues
                for complaint_id in item.evidence_complaint_ids
            }
        )

        return self.success(
            output={
                "analysis": output.model_dump(),
                "dataset_summary": summary,
                "provider": llm_result.provider,
                "deployment": llm_result.deployment,
                "synthetic_data": True,
            },
            confidence=round(confidence, 4),
            latency_ms=total_latency_ms,
            usage=llm_result.usage,
            evidence_ids=evidence_ids,
            warnings=list(
                summary.get("data_quality", {}).get("warnings", [])
            ),
        )
