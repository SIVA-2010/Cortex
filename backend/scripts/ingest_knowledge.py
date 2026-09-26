from __future__ import annotations

from textwrap import dedent

from backend.config import PROJECT_ROOT
from backend.services.vector_store import (
    get_vector_store_service,
)


KNOWLEDGE_DIRECTORY = (
    PROJECT_ROOT
    / "sample_data"
    / "knowledge_base"
)


SAMPLE_DOCUMENTS = {
    "customer_support_policy.md": """
        # Customer Support Policy

        Synthetic demonstration policy for the CORTEX project.

        Every complaint must receive a unique complaint reference.
        Complaints must be classified by issue type, severity,
        affected product, channel and customer impact.

        Critical complaints include safety risks, privacy incidents,
        security incidents, legal threats and complete service
        outages. Critical complaints must be escalated immediately.

        Support agents must not close a complaint until the recorded
        resolution addresses the reported issue. Repeated customer
        contacts must be linked to the original complaint.

        Important analytical findings must include supporting
        complaint references. Assumptions must be clearly separated
        from verified facts.
    """,
    "service_level_policy.md": """
        # Service Level Policy

        Synthetic demonstration policy for the CORTEX project.

        Critical complaints require an initial response within one
        hour and a mitigation plan within four hours.

        High-severity complaints require an initial response within
        four hours and a resolution target of twenty-four hours.

        Medium-severity complaints require an initial response within
        one business day and a resolution target of three business
        days.

        Low-severity complaints require an initial response within two
        business days and a resolution target of five business days.

        Any missed response or resolution target must be marked as an
        SLA breach and included in operational reporting.
    """,
    "refund_policy.md": """
        # Refund and Compensation Policy

        Synthetic demonstration policy for the CORTEX project.

        A confirmed duplicate charge may receive an automatic refund
        up to 100 currency units.

        Refunds above 100 currency units require reviewer approval.
        Goodwill compensation above 50 currency units also requires
        reviewer approval.

        A refund recommendation must include the complaint reference,
        transaction evidence, refund reason and calculated amount.

        The AI platform may recommend a refund, but it must not execute
        a financial transaction. Financial actions remain under human
        control.
    """,
    "privacy_and_pii_policy.md": """
        # Privacy and PII Policy

        Synthetic demonstration policy for the CORTEX project.

        Personal information includes names, email addresses, phone
        numbers, postal addresses, account numbers and transaction
        identifiers.

        Personal information must be masked before complaint text is
        sent to a language model. Account numbers may display only the
        final four characters when required for verification.

        Raw personal information must not be written to workflow logs,
        evaluation records, vector metadata or executive reports.

        Suspected privacy exposure is classified as high risk and must
        be referred to a human reviewer.
    """,
    "escalation_policy.md": """
        # Complaint Escalation Policy

        Synthetic demonstration policy for the CORTEX project.

        A complaint must be escalated immediately when it involves a
        safety issue, security incident, privacy exposure, legal risk
        or complete service outage.

        Escalation is also required when the same customer contacts
        support three or more times about an unresolved issue, when a
        premium customer remains unresolved for more than twenty-four
        hours, or when an SLA target is breached.

        An emerging operational incident must be raised when twenty or
        more complaints with the same issue occur within two hours.

        High-impact production actions, including rollback, service
        shutdown and access revocation, require human approval.
    """,
    "product_incident_policy.md": """
        # Product Incident and Root-Cause Policy

        Synthetic demonstration policy for the CORTEX project.

        A root cause may be classified as confirmed only when it is
        supported by direct operational evidence such as an incident
        record, system log, release record or verified transaction
        failure.

        A root cause may be classified as probable when multiple
        independent sources show a consistent relationship but direct
        confirmation is incomplete.

        A cause based only on complaint wording must be classified as
        unverified.

        Recommendations must show the supporting evidence, confidence
        level, expected business impact and required approval level.
        Production rollback is a high-risk action requiring human
        approval.
    """,
}


def create_sample_files() -> int:
    KNOWLEDGE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    created_count = 0

    for file_name, content in SAMPLE_DOCUMENTS.items():
        path = KNOWLEDGE_DIRECTORY / file_name

        if not path.exists():
            path.write_text(
                dedent(content).strip() + "\n",
                encoding="utf-8",
            )
            created_count += 1

    return created_count


def main() -> None:
    print("-" * 68)
    print("CORTEX ENTERPRISE KNOWLEDGE INGESTION")
    print("-" * 68)

    created_count = create_sample_files()

    print(
        "Sample files created:",
        created_count,
    )
    print(
        "Knowledge directory :",
        KNOWLEDGE_DIRECTORY,
    )

    print("-" * 68)
    print(
        "Loading local embedding model and ingesting documents..."
    )
    print("-" * 68)

    service = get_vector_store_service()

    results = service.ingest_directory(
        KNOWLEDGE_DIRECTORY,
        document_type="enterprise_policy",
    )

    total_chunks = sum(
        result.chunk_count
        for result in results
    )

    for result in results:
        print(
            f"[OK] {result.title} "
            f"-> {result.chunk_count} chunk(s)"
        )

    print("-" * 68)
    print("Files ingested    :", len(results))
    print("Chunks generated  :", total_chunks)
    print(
        "Collection count  :",
        service.collection_count(),
    )
    print(
        "Collection name   :",
        service.collection.name,
    )
    print(
        "Persistent path   :",
        service.persist_path,
    )
    print("-" * 68)

    query = (
        "When must a customer complaint be escalated?"
    )

    print("Test query:", query)
    print("-" * 68)

    search_results = service.search(
        query,
        n_results=3,
    )

    for index, item in enumerate(
        search_results,
        start=1,
    ):
        print(
            f"{index}. {item.title}"
        )
        print(
            f"   Score: {item.relevance_score:.4f}"
        )
        print(
            f"   Evidence: {item.evidence_id}"
        )
        print(
            f"   Text: {item.text[:220]}..."
        )

    print("-" * 68)
    print(
        "CORTEX enterprise knowledge ingestion completed."
    )
    print("-" * 68)


if __name__ == "__main__":
    main()