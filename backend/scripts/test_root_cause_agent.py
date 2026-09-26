from __future__ import annotations

import json
from pathlib import Path

from backend.agents.complaint_agent import ComplaintIntelligenceAgent
from backend.agents.root_cause_agent import RootCauseAnalysisAgent
from backend.config import PROJECT_ROOT
from backend.scripts.generate_complaints import generate_dataset
from backend.scripts.generate_operational_evidence import generate_all
from backend.services.operational_evidence_processor import (
    OperationalEvidenceProcessor,
)


COMPLAINT_DIRECTORY = PROJECT_ROOT / "sample_data" / "complaints"
OPERATIONS_DIRECTORY = PROJECT_ROOT / "sample_data" / "operations"
COMPLAINT_CSV = COMPLAINT_DIRECTORY / "customer_complaints_100.csv"
COMPLAINT_ANALYSIS = (
    COMPLAINT_DIRECTORY / "customer_complaints_100_analysis.json"
)
ROOT_CAUSE_RESULT = (
    COMPLAINT_DIRECTORY / "customer_complaints_100_root_cause.json"
)


def separator() -> None:
    print("-" * 76)


def load_or_create_complaint_analysis() -> dict:
    if COMPLAINT_ANALYSIS.exists():
        return json.loads(COMPLAINT_ANALYSIS.read_text(encoding="utf-8"))

    if not COMPLAINT_CSV.exists():
        generate_dataset(
            row_count=100,
            output_path=COMPLAINT_CSV,
            seed=2026,
        )

    print("Complaint analysis JSON was not found; running the Complaint Agent...")
    result = ComplaintIntelligenceAgent().run(csv_path=COMPLAINT_CSV)

    if result.status != "completed":
        raise SystemExit(
            "Complaint Intelligence Agent failed: " + "; ".join(result.errors)
        )

    COMPLAINT_ANALYSIS.write_text(
        json.dumps(result.output, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return result.output


def main() -> None:
    separator()
    print("CORTEX ROOT-CAUSE ANALYSIS TEST")
    separator()

    complaint_output = load_or_create_complaint_analysis()

    required_files = [
        OPERATIONS_DIRECTORY / "service_incidents.csv",
        OPERATIONS_DIRECTORY / "product_release_history.csv",
        OPERATIONS_DIRECTORY / "transaction_failures.csv",
        OPERATIONS_DIRECTORY / "support_team_metrics.csv",
    ]

    if not all(path.exists() for path in required_files):
        print("Generating synthetic operational evidence...")
        generate_all(output_directory=OPERATIONS_DIRECTORY, seed=2026)

    processor = OperationalEvidenceProcessor()
    processed = processor.process_directory(OPERATIONS_DIRECTORY)
    summary = processed.summary

    print("Complaint source  :", COMPLAINT_ANALYSIS)
    print("Operations source :", OPERATIONS_DIRECTORY)
    print("Evidence items    :", summary["total_evidence_items"])
    print("Direct evidence   :", summary["direct_evidence_count"])

    separator()
    print("OPERATIONAL SIGNALS")
    separator()

    for item in summary["transaction_signals"]:
        print(
            f"{item['category']:<20} "
            f"pre={item['pre_failure_rate_pct']:.3f}% "
            f"post={item['post_failure_rate_pct']:.3f}% "
            f"ratio={item['failure_rate_ratio']:.2f}x "
            f"evidence={item['evidence_id']}"
        )

    for item in summary["support_signals"]:
        if item["category"] == "Refund Delay":
            print(
                f"Refund support signal  pre backlog="
                f"{item['pre_average_backlog']:.2f}, "
                f"post backlog={item['post_average_backlog']:.2f}, "
                f"evidence={item['evidence_id']}"
            )

    separator()
    print("Running Root-Cause Analysis Agent...")
    separator()

    result = RootCauseAnalysisAgent().run(
        complaint_output=complaint_output,
        operations_directory=OPERATIONS_DIRECTORY,
    )

    print("Agent       :", result.agent_name)
    print("Status      :", result.status)
    print("Confidence  :", result.confidence)
    print("Latency ms  :", result.latency_ms)
    print("Tokens      :", result.total_tokens)
    print("Evidence IDs:", len(result.evidence_ids))

    if result.warnings:
        print("Warnings    :", result.warnings)

    if result.status != "completed":
        print("Errors      :", result.errors)
        raise SystemExit("Root-Cause Analysis Agent test failed.")

    analysis = result.output["analysis"]

    separator()
    print("EXECUTIVE SUMMARY")
    separator()
    print(analysis["executive_summary"])

    separator()
    print("ROOT-CAUSE FINDINGS")
    separator()

    for index, finding in enumerate(analysis["findings"], start=1):
        print(f"{index}. {finding['title']}")
        print("   Category   :", finding["category"])
        print("   Status     :", finding["status"])
        print("   Confidence :", finding["confidence"])
        print("   Cause      :", finding["root_cause"])
        print("   Evidence   :", finding["supporting_evidence_ids"])
        print("   Complaints :", finding["complaint_evidence_ids"])
        print("   Risk       :", finding["risk_level"])
        print("   Approval   :", finding["human_approval_required"])
        print("   Action     :", finding["recommended_action"])
        if finding["evidence_gaps"]:
            print("   Gaps       :", finding["evidence_gaps"])

    ROOT_CAUSE_RESULT.write_text(
        json.dumps(result.output, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    separator()
    print("Result file   :", ROOT_CAUSE_RESULT)
    print("Synthetic data: yes")
    print("CORTEX Root-Cause Analysis Agent completed successfully.")
    separator()


if __name__ == "__main__":
    main()
