from sqlalchemy.orm import configure_mappers

from backend.models import Base


def test_final_schema_metadata_is_complete() -> None:
    configure_mappers()
    required = {
        "users", "agents", "missions", "tasks", "task_dependencies",
        "workflow_runs", "agent_runs", "claims", "evidence",
        "governance_decisions", "evaluation_results", "workflow_approvals",
        "uploaded_documents", "audit_logs", "shadow_runs",
    }
    assert required.issubset(Base.metadata.tables)
