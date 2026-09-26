from __future__ import annotations

import importlib
from pathlib import Path

from sqlalchemy import text

from backend.config import PROJECT_ROOT, get_settings
from backend.database import engine


REQUIRED_MODULES = [
    "backend.services.llm",
    "backend.services.vector_store",
    "backend.agents.base_agent",
    "backend.agents.knowledge_agent",
    "backend.agents.complaint_agent",
    "backend.agents.root_cause_agent",
    "backend.agents.verification_agent",
    "backend.agents.guardian_agent",
    "backend.agents.quality_evaluator",
    "backend.agents.report_agent",
    "backend.graph.workflow",
]

REQUIRED_PATHS = [
    PROJECT_ROOT / "sample_data" / "complaints" / "customer_complaints_100.csv",
    PROJECT_ROOT / "sample_data" / "operations",
    PROJECT_ROOT / "sample_data" / "knowledge_base",
]


def main() -> None:
    print("-" * 72)
    print("CORTEX FINAL PREFLIGHT")
    print("-" * 72)
    for module_name in REQUIRED_MODULES:
        importlib.import_module(module_name)
        print(f"[OK] {module_name}")

    for path in REQUIRED_PATHS:
        if not path.exists():
            raise SystemExit(f"[MISSING] {path}")
        print(f"[OK] {path.relative_to(PROJECT_ROOT)}")

    settings = get_settings()
    if settings.llm_provider == "azure":
        if not settings.azure_openai_endpoint or not settings.azure_api_key_value:
            raise SystemExit("Azure OpenAI configuration is incomplete.")

    with engine.connect() as connection:
        database_name, user_name = connection.execute(
            text("SELECT current_database(), current_user")
        ).one()
    print(f"[OK] PostgreSQL database={database_name} user={user_name}")
    print(f"[OK] LLM provider={settings.llm_provider}")
    print(f"[OK] Checkpointer={settings.workflow_checkpointer}")
    print("-" * 72)
    print("CORTEX final preflight passed.")


if __name__ == "__main__":
    main()
