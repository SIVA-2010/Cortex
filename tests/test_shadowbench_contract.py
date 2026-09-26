from backend.services.shadowbench_service import ShadowBenchService


def test_shadowbench_extracts_evidence_and_categories() -> None:
    report = {
        "key_findings": [
            {"category": "Payment Failure", "evidence_ids": ["INC-1", "REL-1"]},
            {"category": "Refund Delay", "evidence_ids": ["OPS-1"]},
        ]
    }
    assert ShadowBenchService._evidence_ids(report) == {"INC-1", "REL-1", "OPS-1"}
    assert ShadowBenchService._categories(report) == {"payment failure", "refund delay"}
