from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_FILES: dict[str, str] = {
    "service_incidents": "service_incidents.csv",
    "product_release_history": "product_release_history.csv",
    "transaction_failures": "transaction_failures.csv",
    "support_team_metrics": "support_team_metrics.csv",
}

REQUIRED_COLUMNS: dict[str, set[str]] = {
    "service_incidents": {
        "incident_id",
        "started_at",
        "ended_at",
        "product",
        "component",
        "region",
        "severity",
        "incident_type",
        "description",
        "confirmed_root_cause",
        "status",
        "affected_version",
        "related_category",
    },
    "product_release_history": {
        "release_id",
        "released_at",
        "product",
        "version",
        "component",
        "change_summary",
        "risk_level",
        "rollback_available",
        "related_incident_id",
        "validation_status",
        "related_category",
    },
    "transaction_failures": {
        "event_date",
        "product",
        "version",
        "failure_type",
        "failure_count",
        "total_attempts",
        "failure_rate_pct",
        "region",
        "related_incident_id",
        "related_category",
    },
    "support_team_metrics": {
        "metric_date",
        "team",
        "region",
        "open_backlog",
        "average_resolution_hours",
        "sla_breach_rate_pct",
        "available_agents",
        "absence_rate_pct",
        "repeat_contact_rate_pct",
        "related_category",
    },
}

CATEGORY_CHANGE_DATES: dict[str, str] = {
    "Payment Failure": "2026-07-18",
    "Login & Access": "2026-07-12",
    "Delivery Tracking": "2026-07-27",
    "Refund Delay": "2026-07-15",
}


@dataclass(slots=True)
class ProcessedOperationalEvidence:
    dataframes: dict[str, pd.DataFrame]
    summary: dict[str, Any]


class OperationalEvidenceProcessor:
    """Validate and summarize synthetic operational evidence for root-cause analysis."""

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file_handle:
            for block in iter(lambda: file_handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")

    @staticmethod
    def _normalize_bool(series: pd.Series) -> pd.Series:
        return (
            series.fillna(False)
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes", "y", "t"})
        )

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None or pd.isna(value):
            return ""
        return str(value).strip()

    @staticmethod
    def _load_csv(path: Path, logical_name: str) -> pd.DataFrame:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Operational evidence file not found: {path}")

        dataframe = pd.read_csv(path)
        dataframe.columns = [
            str(column).strip().lower() for column in dataframe.columns
        ]

        missing = sorted(REQUIRED_COLUMNS[logical_name] - set(dataframe.columns))
        if missing:
            raise ValueError(
                f"{path.name} is missing required columns: " + ", ".join(missing)
            )

        if dataframe.empty:
            raise ValueError(f"{path.name} contains no data rows.")

        return dataframe

    @staticmethod
    def _incident_strength(row: pd.Series) -> str:
        confirmed = OperationalEvidenceProcessor._clean_text(
            row.get("confirmed_root_cause", "")
        )
        status = OperationalEvidenceProcessor._clean_text(
            row.get("status", "")
        ).lower()

        if confirmed and status in {"resolved", "closed"}:
            return "direct"
        if status in {"mitigated", "investigating", "monitoring"}:
            return "corroborating"
        return "temporal"

    @staticmethod
    def _release_strength(row: pd.Series) -> str:
        status = OperationalEvidenceProcessor._clean_text(
            row.get("validation_status", "")
        ).lower()
        related_incident = OperationalEvidenceProcessor._clean_text(
            row.get("related_incident_id", "")
        )

        if related_incident and status in {"failed_post_release", "rolled_back"}:
            return "direct"
        if related_incident:
            return "corroborating"
        return "temporal"

    @staticmethod
    def _failure_signal(
        group: pd.DataFrame,
        *,
        category: str,
        change_date: pd.Timestamp,
    ) -> dict[str, Any]:
        pre_start = change_date - pd.Timedelta(days=7)
        pre = group[
            (group["event_date"] >= pre_start)
            & (group["event_date"] < change_date)
        ]
        post = group[
            (group["event_date"] >= change_date)
            & (group["event_date"] <= change_date + pd.Timedelta(days=6))
        ]

        pre_failures = int(pre["failure_count"].sum())
        post_failures = int(post["failure_count"].sum())
        pre_attempts = int(pre["total_attempts"].sum())
        post_attempts = int(post["total_attempts"].sum())

        pre_rate = (pre_failures / pre_attempts * 100) if pre_attempts else 0.0
        post_rate = (post_failures / post_attempts * 100) if post_attempts else 0.0
        rate_ratio = (post_rate / pre_rate) if pre_rate > 0 else 0.0
        increase_pct = (
            ((post_rate - pre_rate) / pre_rate) * 100 if pre_rate > 0 else 0.0
        )

        evidence_id = f"TXSIG-{OperationalEvidenceProcessor._slug(category).upper()}"
        incident_ids = sorted(
            {
                str(value).strip()
                for value in post["related_incident_id"].fillna("")
                if str(value).strip()
            }
        )

        return {
            "evidence_id": evidence_id,
            "category": category,
            "change_date": change_date.date().isoformat(),
            "pre_7_day_failure_count": pre_failures,
            "post_7_day_failure_count": post_failures,
            "pre_failure_rate_pct": round(pre_rate, 3),
            "post_failure_rate_pct": round(post_rate, 3),
            "failure_rate_ratio": round(rate_ratio, 2),
            "failure_rate_increase_pct": round(increase_pct, 2),
            "related_incident_ids": incident_ids,
            "strength": "statistical" if rate_ratio >= 1.5 else "temporal",
            "description": (
                f"The seven-day failure rate for {category} changed from "
                f"{pre_rate:.3f}% to {post_rate:.3f}% around {change_date.date()}."
            ),
        }

    @staticmethod
    def _support_signals(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
        signals: list[dict[str, Any]] = []

        for category, change_date_value in CATEGORY_CHANGE_DATES.items():
            category_rows = dataframe[dataframe["related_category"] == category]
            if category_rows.empty:
                continue

            change_date = pd.Timestamp(change_date_value, tz="UTC")
            pre = category_rows[
                (category_rows["metric_date"] >= change_date - pd.Timedelta(days=7))
                & (category_rows["metric_date"] < change_date)
            ]
            post = category_rows[
                (category_rows["metric_date"] >= change_date)
                & (category_rows["metric_date"] <= change_date + pd.Timedelta(days=6))
            ]

            if pre.empty or post.empty:
                continue

            pre_backlog = float(pre["open_backlog"].mean())
            post_backlog = float(post["open_backlog"].mean())
            pre_breach = float(pre["sla_breach_rate_pct"].mean())
            post_breach = float(post["sla_breach_rate_pct"].mean())
            pre_agents = float(pre["available_agents"].mean())
            post_agents = float(post["available_agents"].mean())

            evidence_id = f"SUPSIG-{OperationalEvidenceProcessor._slug(category).upper()}"
            strength = (
                "corroborating"
                if post_backlog >= pre_backlog * 1.4
                or post_breach >= pre_breach + 12
                else "temporal"
            )

            signals.append(
                {
                    "evidence_id": evidence_id,
                    "category": category,
                    "change_date": change_date.date().isoformat(),
                    "pre_average_backlog": round(pre_backlog, 2),
                    "post_average_backlog": round(post_backlog, 2),
                    "pre_sla_breach_rate_pct": round(pre_breach, 2),
                    "post_sla_breach_rate_pct": round(post_breach, 2),
                    "pre_available_agents": round(pre_agents, 2),
                    "post_available_agents": round(post_agents, 2),
                    "regions": sorted(
                        {str(value) for value in post["region"].dropna().unique()}
                    ),
                    "teams": sorted(
                        {str(value) for value in post["team"].dropna().unique()}
                    ),
                    "strength": strength,
                    "description": (
                        f"For {category}, average backlog changed from "
                        f"{pre_backlog:.2f} to {post_backlog:.2f}, while the "
                        f"average SLA breach rate changed from {pre_breach:.2f}% "
                        f"to {post_breach:.2f}%."
                    ),
                }
            )

        return signals

    @staticmethod
    def _category_index(evidence_catalog: list[dict[str, Any]]) -> dict[str, Any]:
        index: dict[str, dict[str, Any]] = {}

        for item in evidence_catalog:
            category = str(item.get("category", "Uncategorized"))
            bucket = index.setdefault(
                category,
                {
                    "evidence_ids": [],
                    "direct_evidence_ids": [],
                    "strengths": [],
                },
            )

            evidence_id = str(item.get("evidence_id", ""))
            strength = str(item.get("strength", "temporal"))

            if evidence_id:
                bucket["evidence_ids"].append(evidence_id)
            bucket["strengths"].append(strength)

            if strength == "direct" and evidence_id:
                bucket["direct_evidence_ids"].append(evidence_id)

        for bucket in index.values():
            bucket["evidence_ids"] = sorted(set(bucket["evidence_ids"]))
            bucket["direct_evidence_ids"] = sorted(
                set(bucket["direct_evidence_ids"])
            )
            bucket["strengths"] = sorted(set(bucket["strengths"]))

        return index

    def process_directory(
        self,
        directory: str | Path,
    ) -> ProcessedOperationalEvidence:
        directory_path = Path(directory).resolve()
        if not directory_path.exists() or not directory_path.is_dir():
            raise FileNotFoundError(
                f"Operational evidence directory not found: {directory_path}"
            )

        frames: dict[str, pd.DataFrame] = {}
        source_files: list[dict[str, Any]] = []

        for logical_name, file_name in REQUIRED_FILES.items():
            path = directory_path / file_name
            frame = self._load_csv(path, logical_name)
            frames[logical_name] = frame
            source_files.append(
                {
                    "logical_name": logical_name,
                    "file_name": file_name,
                    "source_path": str(path),
                    "row_count": int(len(frame)),
                    "sha256": self._sha256(path),
                }
            )

        incidents = frames["service_incidents"].copy()
        incidents["started_at"] = pd.to_datetime(
            incidents["started_at"], errors="coerce", utc=True
        )
        incidents["ended_at"] = pd.to_datetime(
            incidents["ended_at"], errors="coerce", utc=True
        )
        if incidents["started_at"].isna().any():
            raise ValueError("service_incidents.csv contains invalid started_at values.")

        releases = frames["product_release_history"].copy()
        releases["released_at"] = pd.to_datetime(
            releases["released_at"], errors="coerce", utc=True
        )
        releases["rollback_available"] = self._normalize_bool(
            releases["rollback_available"]
        )
        if releases["released_at"].isna().any():
            raise ValueError(
                "product_release_history.csv contains invalid released_at values."
            )

        transactions = frames["transaction_failures"].copy()
        transactions["event_date"] = pd.to_datetime(
            transactions["event_date"], errors="coerce", utc=True
        )
        for column in (
            "failure_count",
            "total_attempts",
            "failure_rate_pct",
        ):
            transactions[column] = pd.to_numeric(
                transactions[column], errors="coerce"
            ).fillna(0)
        if transactions["event_date"].isna().any():
            raise ValueError(
                "transaction_failures.csv contains invalid event_date values."
            )

        support = frames["support_team_metrics"].copy()
        support["metric_date"] = pd.to_datetime(
            support["metric_date"], errors="coerce", utc=True
        )
        for column in (
            "open_backlog",
            "average_resolution_hours",
            "sla_breach_rate_pct",
            "available_agents",
            "absence_rate_pct",
            "repeat_contact_rate_pct",
        ):
            support[column] = pd.to_numeric(support[column], errors="coerce").fillna(0)
        if support["metric_date"].isna().any():
            raise ValueError(
                "support_team_metrics.csv contains invalid metric_date values."
            )

        frames = {
            "service_incidents": incidents,
            "product_release_history": releases,
            "transaction_failures": transactions,
            "support_team_metrics": support,
        }

        incident_evidence: list[dict[str, Any]] = []
        for _, row in incidents.sort_values("started_at").iterrows():
            incident_evidence.append(
                {
                    "evidence_id": str(row["incident_id"]),
                    "evidence_type": "service_incident",
                    "category": str(row["related_category"]),
                    "observed_at": row["started_at"].isoformat(),
                    "product": str(row["product"]),
                    "component": str(row["component"]),
                    "severity": str(row["severity"]),
                    "status": str(row["status"]),
                    "description": str(row["description"]),
                    "confirmed_root_cause": self._clean_text(
                        row.get("confirmed_root_cause", "")
                    ),
                    "strength": self._incident_strength(row),
                }
            )

        release_evidence: list[dict[str, Any]] = []
        for _, row in releases.sort_values("released_at").iterrows():
            release_evidence.append(
                {
                    "evidence_id": str(row["release_id"]),
                    "evidence_type": "product_release",
                    "category": str(row["related_category"]),
                    "observed_at": row["released_at"].isoformat(),
                    "product": str(row["product"]),
                    "component": str(row["component"]),
                    "version": str(row["version"]),
                    "change_summary": str(row["change_summary"]),
                    "validation_status": str(row["validation_status"]),
                    "rollback_available": bool(row["rollback_available"]),
                    "related_incident_id": self._clean_text(
                        row.get("related_incident_id", "")
                    ),
                    "strength": self._release_strength(row),
                }
            )

        transaction_signals: list[dict[str, Any]] = []
        for category, change_date_value in CATEGORY_CHANGE_DATES.items():
            group = transactions[transactions["related_category"] == category]
            if group.empty:
                continue
            transaction_signals.append(
                self._failure_signal(
                    group,
                    category=category,
                    change_date=pd.Timestamp(change_date_value, tz="UTC"),
                )
            )

        support_signals = self._support_signals(support)

        evidence_catalog = (
            incident_evidence
            + release_evidence
            + [
                {
                    **item,
                    "evidence_type": "transaction_signal",
                }
                for item in transaction_signals
            ]
            + [
                {
                    **item,
                    "evidence_type": "support_signal",
                }
                for item in support_signals
            ]
        )

        summary = {
            "synthetic_data": True,
            "source_directory": str(directory_path),
            "source_files": source_files,
            "incident_evidence": incident_evidence,
            "release_evidence": release_evidence,
            "transaction_signals": transaction_signals,
            "support_signals": support_signals,
            "evidence_catalog": evidence_catalog,
            "category_evidence_index": self._category_index(evidence_catalog),
            "total_evidence_items": len(evidence_catalog),
            "direct_evidence_count": sum(
                1 for item in evidence_catalog if item.get("strength") == "direct"
            ),
            "limitations": [
                "All operational records are synthetic demonstration evidence.",
                "Temporal overlap alone does not prove causation.",
                "Confirmed findings require at least one direct evidence item.",
            ],
        }

        return ProcessedOperationalEvidence(
            dataframes=frames,
            summary=summary,
        )
