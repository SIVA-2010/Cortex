from __future__ import annotations

import json
from pathlib import Path

from backend.agents.guardian_agent import GuardianGovernanceAgent
from backend.agents.quality_evaluator import QualityEvaluator
from backend.agents.report_agent import ExecutiveReportAgent
from backend.agents.verification_agent import VerificationAgent
from backend.config import PROJECT_ROOT
from backend.services.trust_graph_service import TrustGraphService


COMPLAINT_DIRECTORY = PROJECT_ROOT / "sample_data" / "complaints"
ROOT_CAUSE_CANDIDATES = [
    COMPLAINT_DIRECTORY / "customer_complaints_100_root_cause.json",
    COMPLAINT_DIRECTORY / "customer_complaints_100_root_causes.json",
]
COMPLAINT_ANALYSIS = COMPLAINT_DIRECTORY / "customer_complaints_100_analysis.json"
OUTPUT_PATH = COMPLAINT_DIRECTORY / "customer_complaints_100_decision_assurance.json"


def separator() -> None:
    print("-" * 82)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_root_cause_file() -> Path:
    for path in ROOT_CAUSE_CANDIDATES:
        if path.exists():
            return path
    raise SystemExit(
        "Root-cause output was not found. Run "
        "`python -m backend.scripts.test_root_cause_agent` first."
    )


def main() -> None:
    separator()
    print("CORTEX DECISION ASSURANCE PIPELINE")
    separator()

    root_path = find_root_cause_file()
    root_cause = load_json(root_path)
    complaint_output = load_json(COMPLAINT_ANALYSIS) if COMPLAINT_ANALYSIS.exists() else {}

    print("Root-cause input :", root_path)
    print("Complaint input  :", COMPLAINT_ANALYSIS if COMPLAINT_ANALYSIS.exists() else "not found")

    separator()
    print("1. VERIFICATION AGENT")
    separator()
    verification = VerificationAgent().run(root_cause_result=root_cause)
    if verification.status != "completed":
        raise SystemExit("Verification failed: " + "; ".join(verification.errors))

    verification_data = verification.output["verification"]
    print("Verified     :", verification_data["verified_claim_count"])
    print("Needs review :", verification_data["review_required_count"])
    print("Rejected     :", verification_data["rejected_claim_count"])
    print("Groundedness :", verification_data["overall_groundedness"])
    print("Hallucination:", verification_data["overall_hallucination_risk"])

    separator()
    print("2. TRUSTGRAPH")
    separator()
    trust_graph = TrustGraphService.build(
        root_cause_result=root_cause,
        verification_result=verification.output,
    )
    print("Claims       :", trust_graph.summary["claim_count"])
    print("Evidence     :", trust_graph.summary["evidence_count"])
    print("Nodes        :", len(trust_graph.nodes))
    print("Edges        :", len(trust_graph.edges))

    separator()
    print("3. GUARDIAN GOVERNANCE AGENT")
    separator()
    governance = GuardianGovernanceAgent().run(
        root_cause_result=root_cause,
        verification_result=verification.output,
        complaint_output=complaint_output,
    )
    if governance.status != "completed":
        raise SystemExit("Governance failed: " + "; ".join(governance.errors))

    governance_data = governance.output["governance"]
    print("Decision     :", governance_data["decision"])
    print("Risk         :", governance_data["risk_level"])
    print("Approval     :", governance_data["approval_required"])
    if governance_data.get("approval_reason"):
        print("Reason       :", governance_data["approval_reason"])

    separator()
    print("4. QUALITY EVALUATOR")
    separator()
    total_tokens = verification.total_tokens
    total_latency_ms = verification.latency_ms + governance.latency_ms
    evaluation = QualityEvaluator().run(
        root_cause_result=root_cause,
        verification_result=verification.output,
        governance_result=governance.output,
        total_tokens=total_tokens,
        total_latency_ms=total_latency_ms,
    )
    if evaluation.status != "completed":
        raise SystemExit("Evaluation failed: " + "; ".join(evaluation.errors))

    evaluation_data = evaluation.output["evaluation"]
    print("Score        :", evaluation_data["mission_success_score"])
    print("Status       :", evaluation_data["status"])
    print("Weakest      :", evaluation_data["weakest_component"])

    separator()
    print("5. EXECUTIVE REPORT AGENT")
    separator()
    report = ExecutiveReportAgent().run(
        root_cause_result=root_cause,
        verification_result=verification.output,
        governance_result=governance.output,
        evaluation_result=evaluation.output,
    )
    if report.status != "completed":
        raise SystemExit("Report generation failed: " + "; ".join(report.errors))

    report_data = report.output["report"]
    print(report_data["executive_summary"])
    print("Findings     :", len(report_data["key_findings"]))
    print("Actions      :", len(report_data["recommendations"]))
    print("Next action  :", report_data["next_recommended_action"])

    output = {
        "root_cause": root_cause,
        "verification": verification.output,
        "trust_graph": trust_graph.model_dump(),
        "governance": governance.output,
        "evaluation": evaluation.output,
        "final_report": report.output,
        "pipeline_metrics": {
            "tokens": verification.total_tokens + report.total_tokens,
            "latency_ms": (
                verification.latency_ms
                + governance.latency_ms
                + evaluation.latency_ms
                + report.latency_ms
            ),
        },
        "synthetic_data": True,
    }
    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    separator()
    print("Output saved  :", OUTPUT_PATH)
    print("CORTEX decision assurance pipeline completed successfully.")
    separator()


if __name__ == "__main__":
    main()
