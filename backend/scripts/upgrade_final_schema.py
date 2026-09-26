from __future__ import annotations

from sqlalchemy import inspect

from backend.database import engine
from backend.models import Base


REQUIRED_TABLES = {
    "users",
    "agents",
    "missions",
    "tasks",
    "workflow_runs",
    "agent_runs",
    "claims",
    "evidence",
    "governance_decisions",
    "evaluation_results",
    "workflow_approvals",
    "uploaded_documents",
    "audit_logs",
    "task_dependencies",
    "shadow_runs",
}


def main() -> None:
    print("-" * 72)
    print("CORTEX FINAL SCHEMA UPGRADE")
    print("-" * 72)
    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    tables = set(inspector.get_table_names(schema="public"))
    missing = sorted(REQUIRED_TABLES - tables)

    print("Database tables:")
    for table in sorted(REQUIRED_TABLES & tables):
        print(f"  [OK] {table}")

    if missing:
        print("\nMissing tables:")
        for table in missing:
            print(f"  [MISSING] {table}")
        raise SystemExit("Final schema upgrade did not complete.")

    print("-" * 72)
    print("CORTEX final schema is ready.")
    print("-" * 72)


if __name__ == "__main__":
    main()
