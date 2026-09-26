from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backend.config import Settings, get_settings
from backend.services.pii_service import PIIService, get_pii_service


REQUIRED_COLUMNS = {
    "complaint_id",
    "received_at",
    "complaint_text",
    "channel",
    "region",
    "product",
}

OPTIONAL_DEFAULTS: dict[str, Any] = {
    "product_version": "unknown",
    "priority": "medium",
    "sentiment": "negative",
    "customer_tier": "standard",
    "first_response_minutes": 0,
    "resolution_hours": 0.0,
    "resolution_status": "open",
    "sla_breached": False,
    "repeat_contact_count": 0,
    "escalation_flag": False,
    "refund_amount": 0.0,
    "support_team": "unassigned",
    "pii_present": False,
    "source_system": "unknown",
}

CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "Payment Failure",
        (
            "payment failed",
            "transaction failed",
            "card declined",
            "card was declined",
            "gateway timeout",
            "checkout failed",
            "payment error",
            "money debited",
        ),
    ),
    (
        "Login & Access",
        (
            "cannot login",
            "can't login",
            "unable to login",
            "sign in",
            "otp",
            "password reset",
            "account locked",
            "token refresh",
            "authentication",
        ),
    ),
    (
        "Refund Delay",
        (
            "refund delayed",
            "refund pending",
            "refund not received",
            "refund was approved but not received",
            "waiting for refund",
            "refund",
            "money back",
            "reimbursement",
        ),
    ),
    (
        "Delivery Tracking",
        (
            "delivery status",
            "tracking status",
            "shipment",
            "courier",
            "parcel",
            "not delivered",
            "delivery delayed",
        ),
    ),
    (
        "Billing & Charges",
        (
            "charged twice",
            "duplicate charge",
            "incorrect charge",
            "billing error",
            "unexpected fee",
            "invoice",
        ),
    ),
    (
        "Service Outage",
        (
            "service outage",
            "service down",
            "service is down",
            "system unavailable",
            "application has been unavailable",
            "complete outage",
            "downtime",
        ),
    ),
    (
        "Privacy & Data",
        (
            "privacy exposure",
            "data leak",
            "personal information",
            "unauthorized access",
            "security incident",
        ),
    ),
    (
        "Product Defect",
        (
            "app crash",
            "application crash",
            "screen freezes",
            "software bug",
            "broken feature",
            "unexpected error",
        ),
    ),
]

PRIORITY_SCORE = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


@dataclass(slots=True)
class ProcessedComplaintData:
    dataframe: pd.DataFrame
    summary: dict[str, Any]


class ComplaintProcessor:
    """Validate, mask and summarize complaint CSV data before LLM use."""

    def __init__(
        self,
        settings: Settings | None = None,
        pii_service: PIIService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.pii_service = pii_service or get_pii_service()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file_handle:
            for block in iter(lambda: file_handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _normalize_boolean(series: pd.Series) -> pd.Series:
        true_values = {"true", "1", "yes", "y", "t"}
        return (
            series.fillna(False)
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(true_values)
        )

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default

        if math.isnan(number) or math.isinf(number):
            return default
        return number

    @staticmethod
    def _top_values(
        dataframe: pd.DataFrame,
        column: str,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        counts = dataframe[column].fillna("unknown").astype(str).value_counts()
        total = max(len(dataframe), 1)

        return [
            {
                "name": str(name),
                "count": int(count),
                "percentage": round((int(count) / total) * 100, 2),
            }
            for name, count in counts.head(limit).items()
        ]

    @staticmethod
    def _infer_category(text: str) -> str:
        normalized = str(text or "").lower()

        for category, phrases in CATEGORY_RULES:
            if any(phrase in normalized for phrase in phrases):
                return category

        return "Other"

    @staticmethod
    def _severity_label(row: pd.Series) -> str:
        priority = str(row.get("priority", "medium")).strip().lower()
        if priority in PRIORITY_SCORE:
            return priority

        text = str(row.get("complaint_text", "")).lower()
        if any(
            phrase in text
            for phrase in (
                "security incident",
                "privacy exposure",
                "complete outage",
                "legal action",
                "safety issue",
            )
        ):
            return "critical"

        return "medium"

    @staticmethod
    def _category_metrics(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
        total = max(len(dataframe), 1)
        rows: list[dict[str, Any]] = []

        for category, group in dataframe.groupby(
            "derived_category",
            dropna=False,
            sort=False,
        ):
            count = len(group)
            rows.append(
                {
                    "category": str(category),
                    "count": int(count),
                    "percentage": round((count / total) * 100, 2),
                    "sla_breach_rate": round(
                        float(group["sla_breached"].mean()) * 100,
                        2,
                    ),
                    "escalation_rate": round(
                        float(group["escalation_flag"].mean()) * 100,
                        2,
                    ),
                    "average_resolution_hours": round(
                        float(group["resolution_hours"].mean()),
                        2,
                    ),
                    "average_repeat_contacts": round(
                        float(group["repeat_contact_count"].mean()),
                        2,
                    ),
                    "refund_total": round(
                        float(group["refund_amount"].sum()),
                        2,
                    ),
                    "critical_count": int(
                        (group["derived_severity"] == "critical").sum()
                    ),
                    "high_or_critical_count": int(
                        group["derived_severity"].isin(
                            ["high", "critical"]
                        ).sum()
                    ),
                }
            )

        return sorted(
            rows,
            key=lambda item: (
                item["count"],
                item["sla_breach_rate"],
            ),
            reverse=True,
        )

    @staticmethod
    def _daily_volume(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
        daily = (
            dataframe.assign(
                received_date=dataframe["received_at"].dt.strftime("%Y-%m-%d")
            )
            .groupby("received_date", dropna=False)
            .size()
            .sort_index()
        )

        return [
            {"date": str(date), "count": int(count)}
            for date, count in daily.items()
        ]

    @staticmethod
    def _emerging_issues(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
        if dataframe.empty:
            return []

        max_date = dataframe["received_at"].max()
        recent_start = max_date - pd.Timedelta(days=6)
        previous_start = max_date - pd.Timedelta(days=13)

        recent = dataframe[dataframe["received_at"] >= recent_start]
        previous = dataframe[
            (dataframe["received_at"] >= previous_start)
            & (dataframe["received_at"] < recent_start)
        ]

        recent_counts = recent["derived_category"].value_counts()
        previous_counts = previous["derived_category"].value_counts()

        items: list[dict[str, Any]] = []
        categories = sorted(
            set(recent_counts.index).union(previous_counts.index)
        )

        for category in categories:
            recent_count = int(recent_counts.get(category, 0))
            previous_count = int(previous_counts.get(category, 0))

            if recent_count < 2:
                continue

            growth_percent = round(
                ((recent_count - previous_count) / max(previous_count, 1))
                * 100,
                2,
            )

            if growth_percent <= 0:
                continue

            category_rows = recent[
                recent["derived_category"] == category
            ]

            items.append(
                {
                    "category": str(category),
                    "recent_7_day_count": recent_count,
                    "previous_7_day_count": previous_count,
                    "growth_percent": growth_percent,
                    "sla_breach_rate": round(
                        float(category_rows["sla_breached"].mean()) * 100,
                        2,
                    ),
                    "sample_complaint_ids": category_rows[
                        "complaint_id"
                    ]
                    .astype(str)
                    .head(5)
                    .tolist(),
                }
            )

        return sorted(
            items,
            key=lambda item: (
                item["growth_percent"],
                item["recent_7_day_count"],
            ),
            reverse=True,
        )[:5]

    @staticmethod
    def _representative_samples(
        dataframe: pd.DataFrame,
        per_category: int = 2,
    ) -> list[dict[str, Any]]:
        ranked = dataframe.sort_values(
            by=[
                "priority_score",
                "sla_breached",
                "repeat_contact_count",
                "received_at",
            ],
            ascending=[False, False, False, False],
        )

        selected = ranked.groupby(
            "derived_category",
            dropna=False,
            sort=False,
        ).head(per_category)

        samples: list[dict[str, Any]] = []
        for _, row in selected.iterrows():
            samples.append(
                {
                    "complaint_id": str(row["complaint_id"]),
                    "received_at": row["received_at"].isoformat(),
                    "category": str(row["derived_category"]),
                    "severity": str(row["derived_severity"]),
                    "channel": str(row["channel"]),
                    "region": str(row["region"]),
                    "product": str(row["product"]),
                    "product_version": str(row["product_version"]),
                    "sla_breached": bool(row["sla_breached"]),
                    "repeat_contact_count": int(
                        row["repeat_contact_count"]
                    ),
                    "masked_complaint_text": str(
                        row["masked_complaint_text"]
                    )[:900],
                }
            )

        return samples

    @staticmethod
    def _decision_signals(
        category_metrics: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not category_metrics:
            return {}

        highest_volume = max(
            category_metrics,
            key=lambda item: item["count"],
        )
        highest_sla = max(
            category_metrics,
            key=lambda item: item["sla_breach_rate"],
        )
        highest_escalation = max(
            category_metrics,
            key=lambda item: item["escalation_rate"],
        )
        highest_refund = max(
            category_metrics,
            key=lambda item: item["refund_total"],
        )

        return {
            "highest_volume_category": highest_volume,
            "highest_sla_risk_category": highest_sla,
            "highest_escalation_category": highest_escalation,
            "highest_refund_cost_category": highest_refund,
        }

    @classmethod
    def validate_dataset_suitability(
        cls,
        path: str | Path,
        *,
        objective: str = "",
    ) -> dict[str, Any]:
        """Validate that a CSV is structurally and semantically suitable for complaint missions.

        This is a deterministic pre-execution gate. It deliberately performs no LLM calls.
        The deeper processing pipeline still performs its existing masking, cleaning and
        summarisation after this gate passes.
        """
        file_path = Path(path).resolve()
        if not file_path.exists() or not file_path.is_file():
            raise ValueError("Dataset unavailable — the uploaded complaint CSV could not be found.")
        if file_path.suffix.lower() != ".csv":
            raise ValueError("Dataset Mismatch — Customer Complaint Intelligence requires a CSV file.")

        try:
            dataframe = pd.read_csv(file_path)
        except Exception as exc:
            raise ValueError(
                f"Dataset Mismatch — the uploaded file could not be parsed as complaint CSV data: {exc}"
            ) from exc

        dataframe.columns = [str(column).strip().lower() for column in dataframe.columns]
        columns = set(dataframe.columns)
        missing_columns = sorted(REQUIRED_COLUMNS - columns)
        logistics_signature = {
            "order_id",
            "customer_id",
            "sku",
            "qty",
            "order_date",
            "promised_date",
            "ship_to_region",
            "status",
            "weight_kg",
            "carrier_id",
            "carrier_name",
            "eta_days",
        }
        logistics_hits = sorted(columns & logistics_signature)

        if missing_columns:
            domain_hint = (
                " The uploaded CSV appears to contain logistics/order data."
                if len(logistics_hits) >= 3
                else ""
            )
            raise ValueError(
                "Dataset Mismatch — the uploaded CSV does not match Customer Complaint Intelligence."
                + domain_hint
                + " Missing required complaint columns: "
                + ", ".join(missing_columns)
                + "."
            )

        row_count = len(dataframe)
        if row_count == 0:
            raise ValueError(
                "Dataset Mismatch — the complaint CSV contains no data rows. Execution has been blocked."
            )
        if row_count > 100_000:
            raise ValueError(
                "Dataset Mismatch — this prototype accepts at most 100,000 complaint rows per run."
            )

        ids = dataframe["complaint_id"].fillna("").astype(str).str.strip()
        texts = dataframe["complaint_text"].fillna("").astype(str).str.strip()
        dates = pd.to_datetime(dataframe["received_at"], errors="coerce", utc=True)
        valid_mask = ids.ne("") & texts.ne("") & dates.notna()
        valid_count = int(valid_mask.sum())
        valid_ratio = valid_count / max(row_count, 1)
        if valid_count == 0 or valid_ratio < 0.50:
            raise ValueError(
                "Dataset Mismatch — fewer than half of the rows contain a usable complaint_id, "
                "complaint_text and received_at value. Execution has been blocked."
            )

        valid_texts = texts[valid_mask].str.lower()
        generic_complaint_terms = (
            "complaint",
            "issue",
            "problem",
            "failed",
            "failure",
            "error",
            "unable",
            "cannot",
            "can't",
            "refund",
            "payment",
            "login",
            "access",
            "delivery",
            "tracking",
            "delay",
            "delayed",
            "charged",
            "outage",
            "support",
            "service",
            "crash",
            "bug",
            "privacy",
            "account",
        )
        keyword_hits = valid_texts.map(
            lambda value: any(term in value for term in generic_complaint_terms)
        )
        keyword_hit_rate = float(keyword_hits.mean()) if len(keyword_hits) else 0.0

        recognised = valid_texts.map(lambda value: cls._infer_category(value) != "Other")
        recognised_category_rate = float(recognised.mean()) if len(recognised) else 0.0
        average_text_length = float(valid_texts.str.len().mean()) if len(valid_texts) else 0.0

        # Require actual complaint-like free text, not just renamed columns. The threshold is
        # intentionally modest so new complaint categories are not rejected merely because
        # they are absent from the demo taxonomy.
        if average_text_length < 12 or max(keyword_hit_rate, recognised_category_rate) < 0.10:
            raise ValueError(
                "Dataset Mismatch — the required complaint columns are present, but the row "
                "content does not look like customer complaint/support issue data. Execution "
                "has been blocked before any agents or model calls run."
            )

        duplicate_rate = float(ids[valid_mask].duplicated(keep=False).mean()) if valid_count else 0.0
        if duplicate_rate > 0.50:
            raise ValueError(
                "Dataset Mismatch — more than half of the usable complaint rows reuse complaint_id values. "
                "Provide a complaint dataset with stable row identifiers."
            )

        # Lightweight objective-to-data alignment. If the mission explicitly names a
        # month/year (for example "July"), require the uploaded complaint dates to
        # overlap that period. This catches a valid complaint CSV that belongs to the
        # wrong reporting period without using an LLM.
        objective_text = str(objective or "").strip().lower()
        month_names = {
            "january": 1, "jan": 1,
            "february": 2, "feb": 2,
            "march": 3, "mar": 3,
            "april": 4, "apr": 4,
            "may": 5,
            "june": 6, "jun": 6,
            "july": 7, "jul": 7,
            "august": 8, "aug": 8,
            "september": 9, "sep": 9, "sept": 9,
            "october": 10, "oct": 10,
            "november": 11, "nov": 11,
            "december": 12, "dec": 12,
        }
        requested_months = {
            month_number
            for month_name, month_number in month_names.items()
            if re.search(rf"\b{re.escape(month_name)}\b", objective_text)
        }
        requested_years = {int(value) for value in re.findall(r"\b20\d{2}\b", objective_text)}
        valid_dates = dates[valid_mask]
        if requested_months or requested_years:
            aligned = pd.Series(True, index=valid_dates.index)
            if requested_months:
                aligned &= valid_dates.dt.month.isin(requested_months)
            if requested_years:
                aligned &= valid_dates.dt.year.isin(requested_years)
            if not bool(aligned.any()):
                date_min = valid_dates.min().date().isoformat()
                date_max = valid_dates.max().date().isoformat()
                requested_period = " / ".join(
                    [
                        ", ".join(sorted({name.title() for name, number in month_names.items() if number in requested_months and len(name) > 3}))
                        if requested_months else "",
                        ", ".join(str(year) for year in sorted(requested_years)) if requested_years else "",
                    ]
                ).strip(" / ")
                raise ValueError(
                    "Dataset Mismatch — the complaint CSV does not overlap the period requested by the mission "
                    f"({requested_period}). Uploaded complaint dates span {date_min} to {date_max}. "
                    "Execution has been blocked before agent execution."
                )

        return {
            "status": "passed",
            "workflow_profile": "complaint",
            "row_count": int(row_count),
            "usable_row_count": valid_count,
            "usable_row_ratio": round(valid_ratio, 4),
            "keyword_hit_rate": round(keyword_hit_rate, 4),
            "recognised_category_rate": round(recognised_category_rate, 4),
            "average_text_length": round(average_text_length, 2),
            "required_columns": sorted(REQUIRED_COLUMNS),
            "validation": "schema_and_semantic",
        }

    def process_csv(
        self,
        path: str | Path,
        *,
        representative_samples_per_category: int = 2,
    ) -> ProcessedComplaintData:
        file_path = Path(path).resolve()

        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(
                f"Complaint CSV was not found: {file_path}"
            )

        if file_path.suffix.lower() != ".csv":
            raise ValueError("Complaint input must be a CSV file.")

        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.settings.max_upload_size_mb:
            raise ValueError(
                f"CSV size {file_size_mb:.2f} MB exceeds the configured "
                f"limit of {self.settings.max_upload_size_mb} MB."
            )

        dataframe = pd.read_csv(file_path)
        dataframe.columns = [
            str(column).strip().lower() for column in dataframe.columns
        ]

        missing_columns = sorted(REQUIRED_COLUMNS - set(dataframe.columns))
        if missing_columns:
            raise ValueError(
                "Complaint CSV is missing required columns: "
                + ", ".join(missing_columns)
            )

        original_row_count = len(dataframe)
        if original_row_count == 0:
            raise ValueError("Complaint CSV contains no data rows.")

        if original_row_count > 100_000:
            raise ValueError(
                "This prototype accepts at most 100,000 complaint rows per run."
            )

        for column, default_value in OPTIONAL_DEFAULTS.items():
            if column not in dataframe.columns:
                dataframe[column] = default_value

        dataframe["complaint_id"] = (
            dataframe["complaint_id"].fillna("").astype(str).str.strip()
        )
        dataframe["complaint_text"] = (
            dataframe["complaint_text"].fillna("").astype(str).str.strip()
        )

        empty_id_count = int((dataframe["complaint_id"] == "").sum())
        empty_text_count = int((dataframe["complaint_text"] == "").sum())

        dataframe = dataframe[
            (dataframe["complaint_id"] != "")
            & (dataframe["complaint_text"] != "")
        ].copy()

        duplicate_count = int(
            dataframe.duplicated(subset=["complaint_id"], keep="first").sum()
        )
        dataframe = dataframe.drop_duplicates(
            subset=["complaint_id"],
            keep="first",
        ).copy()

        parsed_dates = pd.to_datetime(
            dataframe["received_at"],
            errors="coerce",
            utc=True,
        )
        invalid_date_count = int(parsed_dates.isna().sum())
        dataframe["received_at"] = parsed_dates
        dataframe = dataframe[dataframe["received_at"].notna()].copy()

        if dataframe.empty:
            raise ValueError(
                "No valid complaint rows remained after validation."
            )

        string_columns = [
            "channel",
            "region",
            "product",
            "product_version",
            "priority",
            "sentiment",
            "customer_tier",
            "resolution_status",
            "support_team",
            "source_system",
        ]
        for column in string_columns:
            dataframe[column] = (
                dataframe[column]
                .fillna(OPTIONAL_DEFAULTS.get(column, "unknown"))
                .astype(str)
                .str.strip()
            )

        numeric_columns = [
            "first_response_minutes",
            "resolution_hours",
            "repeat_contact_count",
            "refund_amount",
        ]
        for column in numeric_columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            ).fillna(0)

        dataframe["repeat_contact_count"] = dataframe[
            "repeat_contact_count"
        ].astype(int)

        boolean_columns = [
            "sla_breached",
            "escalation_flag",
            "pii_present",
        ]
        for column in boolean_columns:
            dataframe[column] = self._normalize_boolean(dataframe[column])

        masking_results = dataframe["complaint_text"].map(
            self.pii_service.mask_text
        )
        dataframe["masked_complaint_text"] = masking_results.map(
            lambda item: item.masked_text
        )
        dataframe["pii_detection_count"] = masking_results.map(
            lambda item: item.detection_count
        )
        dataframe["detected_pii_types"] = masking_results.map(
            lambda item: ",".join(item.detected_types)
        )
        dataframe["pii_present"] = (
            dataframe["pii_present"]
            | (dataframe["pii_detection_count"] > 0)
        )

        dataframe["derived_category"] = dataframe[
            "masked_complaint_text"
        ].map(self._infer_category)
        dataframe["derived_severity"] = dataframe.apply(
            self._severity_label,
            axis=1,
        )
        dataframe["priority_score"] = dataframe["derived_severity"].map(
            PRIORITY_SCORE
        ).fillna(2).astype(int)

        category_metrics = self._category_metrics(dataframe)
        emerging_issues = self._emerging_issues(dataframe)

        detected_type_counts: dict[str, int] = {}
        for value in dataframe["detected_pii_types"]:
            for pii_type in filter(None, str(value).split(",")):
                detected_type_counts[pii_type] = (
                    detected_type_counts.get(pii_type, 0) + 1
                )

        warnings: list[str] = []
        if duplicate_count:
            warnings.append(
                f"Removed {duplicate_count} duplicate complaint ID(s)."
            )
        if empty_id_count:
            warnings.append(
                f"Removed {empty_id_count} row(s) with empty complaint IDs."
            )
        if empty_text_count:
            warnings.append(
                f"Removed {empty_text_count} row(s) with empty complaint text."
            )
        if invalid_date_count:
            warnings.append(
                f"Removed {invalid_date_count} row(s) with invalid dates."
            )
        if (dataframe["derived_category"] == "Other").mean() > 0.20:
            warnings.append(
                "More than 20% of complaints were classified as Other; "
                "the taxonomy may need refinement."
            )

        summary = {
            "source_file": file_path.name,
            "source_path": str(file_path),
            "file_sha256": self._sha256(file_path),
            "file_size_mb": round(file_size_mb, 4),
            "original_row_count": int(original_row_count),
            "valid_row_count": int(len(dataframe)),
            "date_range": {
                "start": dataframe["received_at"].min().isoformat(),
                "end": dataframe["received_at"].max().isoformat(),
            },
            "category_metrics": category_metrics,
            "channel_distribution": self._top_values(
                dataframe, "channel"
            ),
            "region_distribution": self._top_values(
                dataframe, "region"
            ),
            "product_distribution": self._top_values(
                dataframe, "product"
            ),
            "priority_distribution": self._top_values(
                dataframe, "derived_severity"
            ),
            "resolution_status_distribution": self._top_values(
                dataframe, "resolution_status"
            ),
            "daily_volume": self._daily_volume(dataframe),
            "emerging_issues": emerging_issues,
            "overall_metrics": {
                "sla_breach_count": int(dataframe["sla_breached"].sum()),
                "sla_breach_rate": round(
                    float(dataframe["sla_breached"].mean()) * 100,
                    2,
                ),
                "escalated_count": int(
                    dataframe["escalation_flag"].sum()
                ),
                "escalation_rate": round(
                    float(dataframe["escalation_flag"].mean()) * 100,
                    2,
                ),
                "average_first_response_minutes": round(
                    float(dataframe["first_response_minutes"].mean()),
                    2,
                ),
                "average_resolution_hours": round(
                    float(dataframe["resolution_hours"].mean()),
                    2,
                ),
                "average_repeat_contacts": round(
                    float(dataframe["repeat_contact_count"].mean()),
                    2,
                ),
                "total_refund_amount": round(
                    float(dataframe["refund_amount"].sum()),
                    2,
                ),
                "open_complaints": int(
                    dataframe["resolution_status"]
                    .str.lower()
                    .isin(["open", "pending", "escalated"])
                    .sum()
                ),
            },
            "pii_summary": {
                "rows_with_pii": int(dataframe["pii_present"].sum()),
                "rows_with_detected_pii": int(
                    (dataframe["pii_detection_count"] > 0).sum()
                ),
                "total_detected_items": int(
                    dataframe["pii_detection_count"].sum()
                ),
                "detected_types": dict(
                    sorted(detected_type_counts.items())
                ),
                "masking_applied": True,
            },
            "decision_signals": self._decision_signals(category_metrics),
            "representative_samples": self._representative_samples(
                dataframe,
                per_category=representative_samples_per_category,
            ),
            "data_quality": {
                "duplicate_rows_removed": duplicate_count,
                "empty_ids_removed": empty_id_count,
                "empty_text_rows_removed": empty_text_count,
                "invalid_date_rows_removed": invalid_date_count,
                "warnings": warnings,
            },
        }

        return ProcessedComplaintData(
            dataframe=dataframe.reset_index(drop=True),
            summary=summary,
        )
