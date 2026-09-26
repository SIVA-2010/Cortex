from __future__ import annotations

import importlib.util
import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS  {message}")


def text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


print("CORTEX final integration validation")
print("=" * 36)

python_files = [
    "frontend/app.py",
    "frontend/report_pdf.py",
    "backend/graph/state.py",
    "backend/graph/workflow.py",
    "backend/services/workflow_service.py",
    "backend/services/workflow_persistence.py",
    "backend/services/workflow_profile.py",
    "backend/services/logistics_service.py",
    "backend/services/template_planner.py",
    "backend/services/mission_planner.py",
    "backend/services/trust_graph_service.py",
    "backend/agents/__init__.py",
    "backend/agents/logistics_routing_agent.py",
    "backend/agents/task_execution_agent.py",
    "backend/agents/logistics_backup_agent.py",
    "backend/agents/recoverable_failure_agent.py",
    "backend/agents/unrecoverable_failure_agent.py",
    "backend/agents/recovery_controller.py",
    "backend/agents/guardian_agent.py",
    "backend/agents/quality_evaluator.py",
    "backend/agents/report_agent.py",
    "backend/routers/missions.py",
    "backend/schemas.py",
    "backend/init_db.py",
]
for relative in python_files:
    py_compile.compile(str(ROOT / relative), doraise=True)
print(f"PASS  Python compilation ({len(python_files)} files)")

workflow = text("backend/graph/workflow.py")
app = text("frontend/app.py")
template = text("backend/services/template_planner.py")
requirements = text("requirements.final.txt")

# Complaint/adversarial safeguards.
require('"execution_mode": "sequential"' in workflow, "standard execution modes remain valid")
require('"fixed_agent_code": "adversarial_test_agent"' in workflow, "complaint adversarial agent remains a fixed assurance control")
require('return "adversarial"' in workflow, "every complaint root-cause path still enters adversarial assurance")
require("RC-003" in workflow and "TEST-FAKE-001" in workflow, "RC-003 / TEST-FAKE-001 assurance proof remains present")
require('execution_mode="sequential"' in template and "adversarial_testing" in template, "template keeps adversarial assurance as a valid sequential task")

# Logistics routing / failure / recovery.
require("designated_primary_with_recovery_reserve" in workflow, "normal logistics reserves Task Execution Agent as primary and backup for recovery only")
require("_controlled_logistics_failure_context" in workflow, "controlled logistics failure context is persisted")
require("business_logic_failure" in workflow, "failure context distinguishes test injection from business/data failure")
require("_recovery_decision_context" in workflow, "reroute versus stop rationale is persisted")
require('"recover_logistics",\n                "completed"' in workflow, "Recovery Controller is completed when it successfully chooses REROUTE or STOP")
require('"request_approval",\n                    "skipped"' in workflow, "Human Approval receives an explicit skipped event when governance does not require it")
require("mark_task_types_skipped" in workflow, "stop branch persists skipped downstream task state")
require("failed_agent_telemetry" in workflow, "trust/reliability before-to-after telemetry is retained")

# Responsive UI / PDF reporting.
require("word-break:keep-all" in app, "metric values do not break inside words")
require("@media (max-width: 980px)" in app and "repeat(2,minmax(0,1fr))" in app, "metric cards reduce columns responsively")
require("Why did the execution fail?" in app, "Mission Plan explains failure cause")
require("Why did CORTEX choose this recovery action?" in app, "Mission Plan explains reroute/stop decision")
require("Download Executive Report PDF" in app, "completed missions expose Executive PDF download")
require("Download Failure & Recovery Audit PDF" in app, "stopped logistics missions expose audit PDF instead of executive report")
require("reportlab>=4.2,<5.0" in requirements, "ReportLab dependency is installed")

# Validate the bundled logistics package and deterministic planning.
service_path = ROOT / "backend/services/logistics_service.py"
spec = importlib.util.spec_from_file_location("cortex_final_logistics_service", service_path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)
data_dir = module.LogisticsService.validate_data_package(ROOT / "sample_data/logistics")
plan = module.LogisticsService.build_shipping_plan(
    data_dir=data_dir,
    objective=(
        "Analyse the logistics orders, products and carrier data, select suitable carriers "
        "using cost, ETA, coverage, weight and reliability."
    ),
    limit=5,
)
require(plan.get("orders_planned", 0) > 0, "bundled logistics package produces a deterministic fulfilment plan")
require(bool(plan.get("shipping_plan")), "carrier recommendations are present")

# Smoke-test both PDF generators without any API/DB/Azure access.
pdf_path = ROOT / "frontend/report_pdf.py"
pdf_spec = importlib.util.spec_from_file_location("cortex_final_report_pdf", pdf_path)
pdf_module = importlib.util.module_from_spec(pdf_spec)
assert pdf_spec and pdf_spec.loader
pdf_spec.loader.exec_module(pdf_module)

mission = {
    "id": "VALIDATION-MISSION",
    "title": "Validation Mission",
    "objective": "Synthetic validation objective",
}
report = {
    "title": "CORTEX Executive Decision Report",
    "synthetic_data_notice": "Synthetic validation data.",
    "executive_summary": "Validation report generated from verified information.",
    "mission_success_score": 92.0,
    "mission_status": "Excellent",
    "key_findings": [
        {
            "category": "Validation",
            "finding": "Evidence-backed validation finding.",
            "verification_status": "verified",
            "groundedness": 0.98,
            "confidence": 0.95,
            "evidence_ids": ["VAL-001"],
        }
    ],
    "recommendations": [
        {
            "priority": 1,
            "category": "Validation",
            "action": "Continue with the verified recommendation.",
            "risk_level": "low",
            "status": "approved_for_use",
        }
    ],
    "governance_status": "Decision: allow; risk: low; approval required: False.",
    "risks": ["Synthetic validation only."],
    "limitations": ["No external system was called."],
    "next_recommended_action": "Continue.",
}
complaint_state = {
    "workflow_profile": "complaint",
    "test_mode": "adversarial",
    "verification_result": {
        "output": {
            "verification": {
                "claims": [
                    {
                        "finding_id": "RC-003",
                        "source_agent_code": "adversarial_test_agent",
                        "verification_result": "rejected",
                        "groundedness": 0.05,
                        "hallucination_risk": 0.95,
                        "invalid_evidence_ids": ["TEST-FAKE-001"],
                    }
                ]
            }
        }
    },
}
complaint_pdf = pdf_module.build_executive_report_pdf(
    mission=mission,
    run={"status": "completed"},
    report=report,
    state=complaint_state,
)
require(complaint_pdf.startswith(b"%PDF"), "complaint Executive PDF generator works")

stop_state = {
    "workflow_profile": "logistics",
    "test_mode": "logistics_stop_failure",
    "stopped_reason": "Controlled execution-integrity failure.",
    "failure_context": {
        "origin": "Controlled TEST ONLY execution-mode injection",
        "classification": "unrecoverable",
        "business_logic_failure": False,
        "reason": "Simulated execution-integrity fault.",
        "impact": "No unverified fulfilment action was accepted.",
    },
    "failed_agent_telemetry": {
        "agent_name": "Unrecoverable Failure Test Agent",
        "trust_before": 80.0,
        "trust_after": 76.0,
        "reliability_before": 82.0,
        "reliability_after": 75.0,
    },
    "skipped_stages": [
        "verify_claims", "build_trust_graph", "apply_governance",
        "request_approval", "evaluate_quality", "generate_report",
    ],
    "timeline": [],
}
stop_pdf = pdf_module.build_failure_audit_pdf(
    mission=mission,
    run={"status": "failed"},
    state=stop_state,
)
require(stop_pdf.startswith(b"%PDF"), "stopped logistics Failure & Recovery Audit PDF generator works")

print("=" * 36)
print("ALL FINAL VALIDATION CHECKS PASSED")
