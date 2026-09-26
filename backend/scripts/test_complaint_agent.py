from __future__ import annotations

import json
from pathlib import Path

from backend.agents.complaint_agent import ComplaintIntelligenceAgent
from backend.config import PROJECT_ROOT
from backend.scripts.generate_complaints import generate_dataset
from backend.services.complaint_processor import ComplaintProcessor


COMPLAINT_DIRECTORY = PROJECT_ROOT / "sample_data" / "complaints"
CSV_PATH = COMPLAINT_DIRECTORY / "customer_complaints_100.csv"
RESULT_PATH = COMPLAINT_DIRECTORY / "customer_complaints_100_analysis.json"


def separator() -> None:
    print("-" * 72)


def main() -> None:
    separator()
    print("CORTEX COMPLAINT INTELLIGENCE TEST")
    separator()

    if not CSV_PATH.exists():
        print("Creating synthetic test dataset...")
        generate_dataset(
            row_count=100,
            output_path=CSV_PATH,
            seed=2026,
        )

    processor = ComplaintProcessor()
    processed = processor.process_csv(CSV_PATH)
    summary = processed.summary

    print("Dataset        :", CSV_PATH)
    print("Valid rows     :", summary["valid_row_count"])
    print("Date range     :", summary["date_range"])
    print("SLA breach rate:", summary["overall_metrics"]["sla_breach_rate"])
    print("PII rows       :", summary["pii_summary"]["rows_with_pii"])
    print("PII masked     :", summary["pii_summary"]["masking_applied"])

    separator()
    print("CATEGORY METRICS")
    separator()

    for item in summary["category_metrics"]:
        print(
            f"{item['category']:<22} "
            f"count={item['count']:<4} "
            f"share={item['percentage']:>6.2f}% "
            f"sla={item['sla_breach_rate']:>6.2f}%"
        )

    separator()
    print("Running Complaint Intelligence Agent...")
    separator()

    agent = ComplaintIntelligenceAgent()
    result = agent.run(csv_path=CSV_PATH)

    print("Agent       :", result.agent_name)
    print("Status      :", result.status)
    print("Confidence  :", result.confidence)
    print("Latency ms  :", result.latency_ms)
    print("Tokens      :", result.total_tokens)
    print("Evidence IDs:", len(result.evidence_ids))

    if result.status != "completed":
        print("Errors      :", result.errors)
        raise SystemExit("Complaint Intelligence Agent test failed.")

    analysis = result.output["analysis"]

    separator()
    print("EXECUTIVE SUMMARY")
    separator()
    print(analysis["executive_summary"])

    separator()
    print("RECURRING ISSUES")
    separator()
    for index, item in enumerate(analysis["recurring_issues"], start=1):
        print(f"{index}. {item['issue']}")
        print("   Category :", item["category"])
        print("   Volume   :", item["volume"])
        print("   Impact   :", item["business_impact"])
        print("   Evidence :", item["evidence_complaint_ids"])

    separator()
    print("EMERGING ISSUES")
    separator()
    if analysis["emerging_issues"]:
        for index, item in enumerate(analysis["emerging_issues"], start=1):
            print(f"{index}. {item['issue']}")
            print(
                "   Recent / previous:",
                item["recent_7_day_count"],
                "/",
                item["previous_7_day_count"],
            )
            print("   Growth percent  :", item["growth_percent"])
            print("   Risk            :", item["risk"])
    else:
        print("No statistically useful emerging pattern was detected.")

    COMPLAINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(result.output, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    separator()
    print("Result file   :", RESULT_PATH)
    print("Synthetic data: yes")
    print("CORTEX Complaint Intelligence Agent completed successfully.")
    separator()


if __name__ == "__main__":
    main()
