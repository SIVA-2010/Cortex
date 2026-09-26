from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backend.config import PROJECT_ROOT


DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "sample_data" / "operations"


def _iso(day: int, hour: int = 0, minute: int = 0) -> str:
    value = datetime(2026, 7, day, hour, minute, tzinfo=timezone.utc)
    return value.isoformat()


def generate_service_incidents(output_directory: Path) -> Path:
    rows = [
        {
            "incident_id": "INC-2039",
            "started_at": _iso(12, 8, 10),
            "ended_at": _iso(14, 18, 30),
            "product": "Customer Portal",
            "component": "Authentication Token Service",
            "region": "Global",
            "severity": "high",
            "incident_type": "software_regression",
            "description": (
                "Token refresh failures increased after authentication changes "
                "in Customer Portal version 6.4."
            ),
            "confirmed_root_cause": (
                "A token-cache invalidation defect introduced by release "
                "REL-6.4-AUTH caused repeated sign-in loops."
            ),
            "status": "resolved",
            "affected_version": "6.4",
            "related_category": "Login & Access",
        },
        {
            "incident_id": "INC-2048",
            "started_at": _iso(18, 9, 15),
            "ended_at": _iso(21, 16, 45),
            "product": "Mobile Payments",
            "component": "Payment Gateway",
            "region": "Global",
            "severity": "critical",
            "incident_type": "configuration_failure",
            "description": (
                "Gateway timeouts and failed mobile payments rose sharply after "
                "a production configuration change."
            ),
            "confirmed_root_cause": (
                "The gateway timeout threshold was reduced incorrectly during "
                "release REL-6.4-PAY, causing valid transactions to time out."
            ),
            "status": "resolved",
            "affected_version": "6.4",
            "related_category": "Payment Failure",
        },
        {
            "incident_id": "INC-2052",
            "started_at": _iso(27, 7, 0),
            "ended_at": _iso(31, 22, 0),
            "product": "Order Tracking",
            "component": "Courier Status Synchronization",
            "region": "Global",
            "severity": "medium",
            "incident_type": "integration_delay",
            "description": (
                "Courier-status updates were delayed after a scheduler change, "
                "causing stale delivery information in the customer application."
            ),
            "confirmed_root_cause": "",
            "status": "investigating",
            "affected_version": "5.1",
            "related_category": "Delivery Tracking",
        },
        {
            "incident_id": "INC-2043",
            "started_at": _iso(15, 6, 0),
            "ended_at": _iso(30, 18, 0),
            "product": "Refund Service",
            "component": "Manual Reconciliation Queue",
            "region": "South,West",
            "severity": "high",
            "incident_type": "operational_backlog",
            "description": (
                "Refund cases accumulated in South and West support queues while "
                "staffing capacity was below forecast."
            ),
            "confirmed_root_cause": "",
            "status": "mitigated",
            "affected_version": "3.2",
            "related_category": "Refund Delay",
        },
    ]

    path = output_directory / "service_incidents.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    return path


def generate_release_history(output_directory: Path) -> Path:
    rows = [
        {
            "release_id": "REL-6.4-AUTH",
            "released_at": _iso(12, 7, 30),
            "product": "Customer Portal",
            "version": "6.4",
            "component": "Authentication Token Service",
            "change_summary": (
                "Introduced token-cache and refresh-flow changes for session "
                "performance."
            ),
            "risk_level": "high",
            "rollback_available": True,
            "related_incident_id": "INC-2039",
            "validation_status": "rolled_back",
            "related_category": "Login & Access",
        },
        {
            "release_id": "REL-6.4-PAY",
            "released_at": _iso(18, 8, 0),
            "product": "Mobile Payments",
            "version": "6.4",
            "component": "Payment Gateway",
            "change_summary": (
                "Changed gateway timeout and retry settings to reduce average "
                "checkout latency."
            ),
            "risk_level": "high",
            "rollback_available": True,
            "related_incident_id": "INC-2048",
            "validation_status": "failed_post_release",
            "related_category": "Payment Failure",
        },
        {
            "release_id": "REL-5.1-TRACK",
            "released_at": _iso(26, 21, 0),
            "product": "Order Tracking",
            "version": "5.1",
            "component": "Courier Status Synchronization",
            "change_summary": (
                "Changed courier synchronization scheduling and queue batching."
            ),
            "risk_level": "medium",
            "rollback_available": True,
            "related_incident_id": "INC-2052",
            "validation_status": "under_review",
            "related_category": "Delivery Tracking",
        },
        {
            "release_id": "REL-3.2-REFUND",
            "released_at": _iso(10, 10, 0),
            "product": "Refund Service",
            "version": "3.2",
            "component": "Refund Workflow",
            "change_summary": (
                "Added a new reconciliation review step for higher-value refunds."
            ),
            "risk_level": "medium",
            "rollback_available": False,
            "related_incident_id": "INC-2043",
            "validation_status": "operational_monitoring",
            "related_category": "Refund Delay",
        },
    ]

    path = output_directory / "product_release_history.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    return path


def generate_transaction_failures(
    output_directory: Path,
    seed: int,
) -> Path:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)

    series = [
        {
            "category": "Payment Failure",
            "product": "Mobile Payments",
            "version": "6.4",
            "failure_type": "gateway_timeout",
            "change_day": 18,
            "baseline": (32, 58),
            "spike": (330, 520),
            "post": (95, 180),
            "related_incident_id": "INC-2048",
        },
        {
            "category": "Login & Access",
            "product": "Customer Portal",
            "version": "6.4",
            "failure_type": "token_refresh_failure",
            "change_day": 12,
            "baseline": (18, 42),
            "spike": (145, 245),
            "post": (35, 75),
            "related_incident_id": "INC-2039",
        },
        {
            "category": "Delivery Tracking",
            "product": "Order Tracking",
            "version": "5.1",
            "failure_type": "courier_sync_delay",
            "change_day": 27,
            "baseline": (10, 24),
            "spike": (90, 155),
            "post": (90, 155),
            "related_incident_id": "INC-2052",
        },
    ]

    for day in range(1, 32):
        event_date = start + timedelta(days=day - 1)
        for item in series:
            attempts = rng.randint(2600, 4100)
            change_day = int(item["change_day"])

            if day < change_day:
                low, high = item["baseline"]
            elif day <= min(change_day + 3, 31):
                low, high = item["spike"]
            else:
                low, high = item["post"]

            failures = rng.randint(int(low), int(high))
            failure_rate = round((failures / attempts) * 100, 3)

            rows.append(
                {
                    "event_date": event_date.date().isoformat(),
                    "product": item["product"],
                    "version": item["version"],
                    "failure_type": item["failure_type"],
                    "failure_count": failures,
                    "total_attempts": attempts,
                    "failure_rate_pct": failure_rate,
                    "region": "Global",
                    "related_incident_id": (
                        item["related_incident_id"]
                        if day >= change_day
                        else ""
                    ),
                    "related_category": item["category"],
                }
            )

    path = output_directory / "transaction_failures.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    return path


def generate_support_metrics(
    output_directory: Path,
    seed: int,
) -> Path:
    rng = random.Random(seed + 91)
    rows: list[dict[str, Any]] = []
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)

    combinations = [
        ("Refund Desk", "South", "Refund Delay"),
        ("Refund Desk", "West", "Refund Delay"),
        ("Refund Desk", "North", "Refund Delay"),
        ("Payments Operations", "Global", "Payment Failure"),
        ("Account Services", "Global", "Login & Access"),
        ("Logistics Support", "Global", "Delivery Tracking"),
    ]

    for day in range(1, 32):
        metric_date = start + timedelta(days=day - 1)

        for team, region, category in combinations:
            backlog = rng.randint(15, 42)
            resolution = rng.uniform(8, 28)
            breach = rng.uniform(5, 18)
            available_agents = rng.randint(14, 22)
            absence = rng.uniform(1, 6)
            repeat_rate = rng.uniform(8, 20)

            if category == "Refund Delay" and region in {"South", "West"}:
                if day >= 15:
                    backlog += rng.randint(55, 110)
                    resolution += rng.uniform(32, 65)
                    breach += rng.uniform(30, 48)
                    available_agents = rng.randint(6, 10)
                    absence += rng.uniform(8, 15)
                    repeat_rate += rng.uniform(20, 35)

            if category == "Payment Failure" and 18 <= day <= 22:
                backlog += rng.randint(35, 75)
                resolution += rng.uniform(12, 25)
                breach += rng.uniform(20, 34)
                repeat_rate += rng.uniform(10, 22)

            if category == "Login & Access" and 12 <= day <= 15:
                backlog += rng.randint(25, 55)
                breach += rng.uniform(14, 25)

            if category == "Delivery Tracking" and day >= 27:
                backlog += rng.randint(30, 65)
                resolution += rng.uniform(12, 28)
                breach += rng.uniform(15, 30)

            rows.append(
                {
                    "metric_date": metric_date.date().isoformat(),
                    "team": team,
                    "region": region,
                    "open_backlog": backlog,
                    "average_resolution_hours": round(resolution, 2),
                    "sla_breach_rate_pct": round(min(breach, 99.0), 2),
                    "available_agents": available_agents,
                    "absence_rate_pct": round(min(absence, 45.0), 2),
                    "repeat_contact_rate_pct": round(min(repeat_rate, 95.0), 2),
                    "related_category": category,
                }
            )

    path = output_directory / "support_team_metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    return path


def generate_all(
    *,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    seed: int = 2026,
) -> dict[str, Path]:
    directory = Path(output_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)

    return {
        "service_incidents": generate_service_incidents(directory),
        "product_release_history": generate_release_history(directory),
        "transaction_failures": generate_transaction_failures(directory, seed),
        "support_team_metrics": generate_support_metrics(directory, seed),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic CORTEX operational evidence."
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    paths = generate_all(
        output_directory=arguments.output_directory,
        seed=arguments.seed,
    )

    print("-" * 72)
    print("CORTEX SYNTHETIC OPERATIONAL EVIDENCE")
    print("-" * 72)
    for name, path in paths.items():
        dataframe = pd.read_csv(path)
        print(f"{name:<28}: {len(dataframe):>3} row(s) -> {path}")
    print("-" * 72)
    print("Synthetic data: yes")
    print("Planted evidence:")
    print("- Payment gateway timeout misconfiguration after 18 July")
    print("- Authentication token regression after 12 July")
    print("- Refund backlog and reduced staffing in South and West")
    print("- Delivery tracking synchronization delay after 27 July")
    print("-" * 72)


if __name__ == "__main__":
    main()
