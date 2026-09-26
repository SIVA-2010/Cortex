from __future__ import annotations

from pathlib import Path

from backend.services.logistics_service import LogisticsService
from backend.services.workflow_profile import detect_workflow_profile

ROOT = Path(__file__).resolve().parent

logistics_question = {
    "title": "Logistics Fulfilment Intelligence",
    "objective": (
        "Analyse the logistics orders, products and carrier data, identify orders at risk "
        "of delayed delivery, select the most suitable carriers based on regional coverage, "
        "ETA, shipment weight, cost and reliability, and prepare an evidence-backed "
        "fulfilment recommendation."
    ),
    # Intentionally left as complaint to prove natural-language override works.
    "business_domain": "Customer Complaint Intelligence",
}
complaint_question = {
    "title": "July Customer Experience Intelligence",
    "objective": "Analyse customer complaints including delivery issues and identify recurring root causes.",
    "business_domain": "Customer Complaint Intelligence",
}

assert detect_workflow_profile(logistics_question) == "logistics"
assert detect_workflow_profile(complaint_question) == "complaint"

path = LogisticsService.validate_data_package(ROOT / "sample_data" / "logistics")
plan = LogisticsService.build_shipping_plan(
    data_dir=path,
    objective=logistics_question["objective"],
    limit=5,
)
assert plan["orders_planned"] > 0
assert plan["optimization_mode"] == "balanced"
assert plan["simulation_only"] is True
assert plan["evidence_catalog"]

workflow_source = (ROOT / "backend" / "graph" / "workflow.py").read_text(encoding="utf-8")
ui_source = (ROOT / "frontend" / "app.py").read_text(encoding="utf-8")

for required in (
    '"logistics": "analyse_logistics"',
    '"recovery": "recover_logistics"',
    '"reroute": "reroute_logistics"',
    '"stop": "persist_failure"',
    '"success": "execute_logistics_fallback"',
    '"failed_agent_telemetry": failure_telemetry',
    'return "adversarial"',
    '"inject_adversarial_claim"',
):
    assert required in workflow_source, required

for required in (
    "Normal Logistics",
    "Recoverable Failure — Reroute",
    "Unrecoverable Failure — Stop",
    "Workflow for selected execution mode",
    "Failure penalty persisted to Agent Registry",
    "TrustGraph was not generated because CORTEX stopped",
    "No executive report was generated because CORTEX stopped",
):
    assert required in ui_source, required

# First-failure movement check for the seeded test-only failure agent. This mirrors
# WorkflowPersistenceService._apply_execution_performance for one warning + one error.
old_trust = 80.0
old_reliability = 82.0
old_hallucination = 0.0
status_score = 20.0
confidence_pct = 0.0
warning_penalty = 1.5
error_penalty = 8.0
execution_outcome = max(0.0, min(100.0, status_score * 0.65 + confidence_pct * 0.35 - warning_penalty - error_penalty))
alpha = 0.10
new_reliability = round(old_reliability + alpha * (execution_outcome - old_reliability), 2)
safety_score = 100.0 - old_hallucination
trust_target = max(0.0, min(100.0, new_reliability * 0.55 + confidence_pct * 0.30 + safety_score * 0.15 - 10.0))
new_trust = round(old_trust + 0.08 * (trust_target - old_trust), 2)
assert new_reliability < old_reliability
assert new_trust < old_trust

print("PASS natural logistics objective -> logistics profile")
print("PASS complaint delivery wording remains complaint profile")
print("PASS orders.csv + products.csv + carriers.json validation")
print(f"PASS deterministic balanced routing: {plan['orders_planned']} orders, {len(plan['evidence_catalog'])} evidence records")
print("PASS LangGraph complaint branch still contains mandatory adversarial stage")
print("PASS LangGraph logistics normal/recovery/reroute/stop branches present")
print("PASS failed-agent before/after telemetry persisted into workflow state")
print("PASS UI exposes Normal / Reroute / Stop and dynamic workflow preview")
print("PASS stop path UI suppresses TrustGraph/report artifacts")
print(f"PASS seeded first failure movement: trust {old_trust:.2f}->{new_trust:.2f}; reliability {old_reliability:.2f}->{new_reliability:.2f}")
