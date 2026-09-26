from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backend.config import PROJECT_ROOT


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "sample_data"
    / "complaints"
    / "customer_complaints_100.csv"
)

CHANNELS = ["Email", "Chat", "Call Centre", "Web", "Mobile App"]
REGIONS = ["North", "South", "East", "West", "Central"]
CUSTOMER_TIERS = ["Standard", "Premium", "Enterprise"]
SUPPORT_TEAMS = [
    "Digital Support",
    "Payments Operations",
    "Account Services",
    "Refund Desk",
    "Logistics Support",
]
SOURCE_SYSTEMS = ["CRM", "Email", "Portal", "Chat", "Call Centre"]
NAMES = [
    "Priya Sharma",
    "Arjun Mehta",
    "Nisha Rao",
    "Rahul Verma",
    "Meera Iyer",
    "Karan Singh",
]

CATEGORY_TEMPLATES: dict[str, list[str]] = {
    "Payment Failure": [
        "My payment failed with a gateway timeout, but the money was debited.",
        "The card was declined during checkout even though funds are available.",
        "A transaction failed in the mobile app and I cannot complete the purchase.",
        "The payment error keeps repeating after I press confirm.",
    ],
    "Login & Access": [
        "I cannot login because the OTP never arrives.",
        "The app says account locked after a password reset.",
        "Sign in fails with a token refresh error after the latest update.",
        "Authentication keeps looping and I am unable to login.",
    ],
    "Refund Delay": [
        "My refund is pending and I am still waiting for the money back.",
        "The refund was approved but not received after several days.",
        "Refund delayed even after contacting support repeatedly.",
        "The reimbursement has not reached my account.",
    ],
    "Delivery Tracking": [
        "The delivery status says pending although the courier promised delivery.",
        "Shipment tracking has not updated and the parcel is delayed.",
        "The order shows not delivered but there is no useful tracking status.",
        "Delivery delayed and the mobile app displays the wrong courier status.",
    ],
    "Billing & Charges": [
        "I was charged twice for the same order and need the duplicate charge reversed.",
        "The invoice contains an incorrect charge and an unexpected fee.",
        "A billing error added an amount that I did not authorize.",
        "The statement shows a duplicate charge for one transaction.",
    ],
    "Service Outage": [
        "The service is down and the system is unavailable for all users.",
        "A complete outage is preventing access to the portal.",
        "The application has been unavailable during business hours.",
        "Repeated downtime is blocking all transactions.",
    ],
    "Privacy & Data": [
        "I suspect a privacy exposure because another customer's details appeared.",
        "There may be unauthorized access to my personal information.",
        "A possible data leak exposed account information in the portal.",
        "This security incident needs immediate investigation.",
    ],
    "Product Defect": [
        "The app crashes whenever I open the payment history.",
        "The screen freezes and the feature is unusable.",
        "A software bug causes an unexpected error on every attempt.",
        "The latest version has a broken feature and closes automatically.",
    ],
}

PRODUCT_BY_CATEGORY = {
    "Payment Failure": ("Mobile Payments", "6.4"),
    "Login & Access": ("Customer Portal", "6.4"),
    "Refund Delay": ("Refund Service", "3.2"),
    "Delivery Tracking": ("Order Tracking", "5.1"),
    "Billing & Charges": ("Billing Platform", "4.7"),
    "Service Outage": ("Customer Portal", "6.3"),
    "Privacy & Data": ("Customer Portal", "6.4"),
    "Product Defect": ("Mobile App", "6.4"),
}

PRIORITY_WEIGHTS = {
    "Payment Failure": [0.04, 0.31, 0.52, 0.13],
    "Login & Access": [0.06, 0.50, 0.36, 0.08],
    "Refund Delay": [0.08, 0.52, 0.35, 0.05],
    "Delivery Tracking": [0.09, 0.60, 0.27, 0.04],
    "Billing & Charges": [0.05, 0.44, 0.43, 0.08],
    "Service Outage": [0.01, 0.08, 0.44, 0.47],
    "Privacy & Data": [0.00, 0.05, 0.25, 0.70],
    "Product Defect": [0.05, 0.48, 0.39, 0.08],
}

PRIORITIES = ["low", "medium", "high", "critical"]


def choose_category(rng: random.Random, day: int) -> str:
    categories = list(CATEGORY_TEMPLATES)

    # The dataset intentionally contains explainable patterns for CORTEX:
    # payment failures spike after 18 July, while delivery tracking becomes
    # a smaller but fast-growing issue near the end of the month.
    if day >= 27:
        weights = [0.30, 0.10, 0.11, 0.25, 0.08, 0.05, 0.03, 0.08]
    elif day >= 18:
        weights = [0.42, 0.12, 0.13, 0.06, 0.10, 0.05, 0.03, 0.09]
    elif day >= 12:
        weights = [0.18, 0.27, 0.16, 0.07, 0.12, 0.06, 0.03, 0.11]
    else:
        weights = [0.17, 0.15, 0.18, 0.08, 0.17, 0.08, 0.04, 0.13]

    return rng.choices(categories, weights=weights, k=1)[0]


def choose_priority(rng: random.Random, category: str) -> str:
    return rng.choices(
        PRIORITIES,
        weights=PRIORITY_WEIGHTS[category],
        k=1,
    )[0]


def response_and_resolution(
    rng: random.Random,
    priority: str,
    category: str,
) -> tuple[int, float, bool]:
    response_ranges = {
        "critical": (15, 100),
        "high": (45, 360),
        "medium": (90, 1800),
        "low": (240, 3600),
    }
    resolution_ranges = {
        "critical": (1.0, 8.0),
        "high": (4.0, 42.0),
        "medium": (8.0, 90.0),
        "low": (12.0, 150.0),
    }
    response_limit = {
        "critical": 60,
        "high": 240,
        "medium": 1440,
        "low": 2880,
    }
    resolution_limit = {
        "critical": 4.0,
        "high": 24.0,
        "medium": 72.0,
        "low": 120.0,
    }

    response = rng.randint(*response_ranges[priority])
    resolution = round(rng.uniform(*resolution_ranges[priority]), 2)

    # Refund and payment workloads intentionally have more SLA risk.
    if category in {"Refund Delay", "Payment Failure"} and rng.random() < 0.25:
        response = int(response * rng.uniform(1.4, 2.4))
        resolution = round(resolution * rng.uniform(1.3, 2.1), 2)

    breached = (
        response > response_limit[priority]
        or resolution > resolution_limit[priority]
    )
    return response, resolution, breached


def add_synthetic_pii(
    rng: random.Random,
    complaint_text: str,
    row_number: int,
) -> tuple[str, bool]:
    if rng.random() >= 0.20:
        return complaint_text, False

    name = rng.choice(NAMES)
    email_name = name.lower().replace(" ", ".")
    phone = f"+91 98{rng.randint(100, 999)} {rng.randint(10000, 99999)}"
    account = f"CX{row_number:08d}"
    transaction = f"TXN-{20260700 + row_number}-{rng.randint(1000, 9999)}"

    pii_prefix = (
        f"My name is {name}. Email {email_name}@example.com, "
        f"phone {phone}. Account number {account}. "
        f"Transaction ID {transaction}. "
    )
    return pii_prefix + complaint_text, True


def generate_dataset(
    *,
    row_count: int,
    output_path: str | Path,
    seed: int = 2026,
) -> Path:
    if row_count < 1:
        raise ValueError("row_count must be at least 1.")

    rng = random.Random(seed)
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    rows: list[dict[str, Any]] = []

    for index in range(1, row_count + 1):
        day = rng.randint(1, 31)
        received_at = start + timedelta(
            days=day - 1,
            hours=rng.randint(0, 23),
            minutes=rng.randint(0, 59),
        )

        category = choose_category(rng, day)
        priority = choose_priority(rng, category)
        product, version = PRODUCT_BY_CATEGORY[category]

        complaint_text = rng.choice(CATEGORY_TEMPLATES[category])
        complaint_text, pii_present = add_synthetic_pii(
            rng,
            complaint_text,
            index,
        )

        first_response, resolution_hours, sla_breached = (
            response_and_resolution(rng, priority, category)
        )

        repeat_contacts = rng.choices(
            [0, 1, 2, 3, 4],
            weights=[0.28, 0.34, 0.22, 0.11, 0.05],
            k=1,
        )[0]

        if category == "Refund Delay":
            repeat_contacts = max(repeat_contacts, rng.randint(1, 4))

        escalation_flag = bool(
            priority == "critical"
            or repeat_contacts >= 3
            or sla_breached
        )

        if resolution_hours <= 24 and not escalation_flag:
            resolution_status = "resolved"
        elif escalation_flag and rng.random() < 0.55:
            resolution_status = "escalated"
        else:
            resolution_status = rng.choice(["open", "pending", "resolved"])

        refund_amount = 0.0
        if category in {"Refund Delay", "Billing & Charges"}:
            refund_amount = round(rng.uniform(20, 260), 2)

        region = rng.choice(REGIONS)
        if category == "Refund Delay" and rng.random() < 0.58:
            region = rng.choice(["South", "West"])

        support_team = {
            "Payment Failure": "Payments Operations",
            "Login & Access": "Account Services",
            "Refund Delay": "Refund Desk",
            "Delivery Tracking": "Logistics Support",
        }.get(category, "Digital Support")

        sentiment = rng.choices(
            ["negative", "very_negative", "neutral"],
            weights=[0.62, 0.30, 0.08],
            k=1,
        )[0]

        rows.append(
            {
                "complaint_id": f"CMP-202607-{index:06d}",
                "received_at": received_at.isoformat(),
                "channel": rng.choice(CHANNELS),
                "region": region,
                "product": product,
                "product_version": version,
                "complaint_text": complaint_text,
                "priority": priority,
                "sentiment": sentiment,
                "customer_tier": rng.choices(
                    CUSTOMER_TIERS,
                    weights=[0.70, 0.22, 0.08],
                    k=1,
                )[0],
                "first_response_minutes": first_response,
                "resolution_hours": resolution_hours,
                "resolution_status": resolution_status,
                "sla_breached": sla_breached,
                "repeat_contact_count": repeat_contacts,
                "escalation_flag": escalation_flag,
                "refund_amount": refund_amount,
                "support_team": support_team,
                "pii_present": pii_present,
                "source_system": rng.choice(SOURCE_SYSTEMS),
            }
        )

    dataframe = pd.DataFrame(rows).sort_values("received_at")
    dataframe.to_csv(output, index=False, encoding="utf-8")
    return output


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic CORTEX complaint data."
    )
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    path = generate_dataset(
        row_count=arguments.rows,
        output_path=arguments.output,
        seed=arguments.seed,
    )

    print("-" * 68)
    print("CORTEX SYNTHETIC COMPLAINT DATASET")
    print("-" * 68)
    print("Rows       :", arguments.rows)
    print("Seed       :", arguments.seed)
    print("Output     :", path)
    print("Synthetic  : yes")
    print("Period     : July 2026")
    print("-" * 68)
    print("Planted patterns:")
    print("- Payment failures rise after 18 July")
    print("- Delivery tracking emerges near month end")
    print("- Refund delays concentrate in South and West")
    print("- Some rows include synthetic PII for masking tests")
    print("-" * 68)


if __name__ == "__main__":
    main()
