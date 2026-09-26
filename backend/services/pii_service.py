from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Pattern

from pydantic import BaseModel, Field


class PIIMaskingResult(BaseModel):
    masked_text: str
    detection_count: int = 0
    detected_types: list[str] = Field(default_factory=list)
    counts_by_type: dict[str, int] = Field(default_factory=dict)


class PIIService:
    """Lightweight deterministic PII masking for complaint text.

    The service deliberately stores only detection types and counts. It never
    returns the raw matched values, which helps prevent accidental leakage into
    logs, workflow state, or LLM prompts.
    """

    def __init__(self) -> None:
        self.patterns: list[tuple[str, Pattern[str], str]] = [
            (
                "email",
                re.compile(
                    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
                    re.IGNORECASE,
                ),
                "[EMAIL_REDACTED]",
            ),
            (
                "account_number",
                re.compile(
                    r"\b(?:account|acct|a/c)\s*"
                    r"(?:number|no\.?|#)?\s*[:=-]?\s*"
                    r"[A-Z0-9]{6,24}\b",
                    re.IGNORECASE,
                ),
                "[ACCOUNT_REDACTED]",
            ),
            (
                "transaction_identifier",
                re.compile(
                    r"\b(?:transaction|txn|reference|ref)\s*"
                    r"(?:id|number|no\.?|#)\s*[:=-]?\s*"
                    r"[A-Z0-9][A-Z0-9-]{5,30}\b",
                    re.IGNORECASE,
                ),
                "[TRANSACTION_REDACTED]",
            ),
            (
                "phone",
                re.compile(
                    r"(?<![\w-])(?:\+?\d{1,3}[\s.-]?)?"
                    r"(?:\d{3}[\s.-]?\d{3}[\s.-]?\d{4}"
                    r"|\d{5}[\s.-]?\d{5})(?![\w-])"
                ),
                "[PHONE_REDACTED]",
            ),
            (
                "postal_address",
                re.compile(
                    r"\b\d{1,5}\s+"
                    r"[A-Z][A-Z0-9 .'-]{2,40}\s+"
                    r"(?:STREET|ST|ROAD|RD|AVENUE|AVE|LANE|LN|"
                    r"DRIVE|DR|NAGAR|COLONY)\b",
                    re.IGNORECASE,
                ),
                "[ADDRESS_REDACTED]",
            ),
            (
                "person_name",
                re.compile(
                    r"\b(?:my name is|this is)\s+"
                    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b",
                    re.IGNORECASE,
                ),
                "[NAME_REDACTED]",
            ),
        ]

    def mask_text(self, text: str | None) -> PIIMaskingResult:
        value = str(text or "")
        counts: Counter[str] = Counter()
        masked = value

        for pii_type, pattern, replacement in self.patterns:
            masked, count = pattern.subn(replacement, masked)
            if count:
                counts[pii_type] += count

        return PIIMaskingResult(
            masked_text=masked,
            detection_count=sum(counts.values()),
            detected_types=sorted(counts),
            counts_by_type=dict(sorted(counts.items())),
        )

    def contains_pii(self, text: str | None) -> bool:
        return self.mask_text(text).detection_count > 0


@lru_cache
def get_pii_service() -> PIIService:
    return PIIService()
