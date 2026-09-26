from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import UUID

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.database import SessionLocal
from backend.graph.workflow import CortexWorkflowGraph
from backend.models import Agent, Mission, UploadedDocument, WorkflowRun
from backend.services.complaint_processor import ComplaintProcessor
from backend.services.logistics_service import LogisticsService
from backend.services.workflow_persistence import WorkflowPersistenceService
from backend.services.workflow_profile import detect_workflow_profile


logger = logging.getLogger(__name__)

DATASET_REQUIRED_MESSAGE = (
    "Dataset Required — This mission requires source data before agent execution "
    "can begin. Upload the required dataset to continue."
)


class WorkflowRuntime:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.pool: Any | None = None
        self.checkpointer: Any

        if self.settings.workflow_checkpointer == "postgres":
            try:
                from langgraph.checkpoint.postgres import PostgresSaver
                from psycopg.rows import dict_row
                from psycopg_pool import ConnectionPool
            except ImportError as exc:
                raise RuntimeError(
                    "PostgreSQL LangGraph persistence requires "
                    "langgraph-checkpoint-postgres and psycopg_pool. "
                    "Install requirements.final.txt."
                ) from exc

            self.pool = ConnectionPool(
                conninfo=self.settings.langgraph_database_uri,
                min_size=1,
                max_size=10,
                open=False,
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": 0,
                    "row_factory": dict_row,
                },
            )
            self.pool.open(wait=True)
            self.checkpointer = PostgresSaver(self.pool)
            self.checkpointer.setup()
        else:
            self.checkpointer = InMemorySaver()

        self.workflow = CortexWorkflowGraph(self.checkpointer)

    @property
    def graph(self) -> Any:
        return self.workflow.graph

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "checkpointer": self.settings.workflow_checkpointer,
            "persistent": self.settings.workflow_checkpointer == "postgres",
        }


@lru_cache
def get_workflow_runtime() -> WorkflowRuntime:
    return WorkflowRuntime()


class WorkflowService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.persistence = WorkflowPersistenceService()

    @staticmethod
    def _agent_snapshot(agent: Agent) -> dict[str, Any]:
        return {
            "id": str(agent.id),
            "code": agent.code,
            "name": agent.name,
            "description": agent.description,
            "capabilities": list(agent.capabilities or []),
            "supported_task_types": list(agent.supported_task_types or []),
            "model_provider": agent.model_provider,
            "model_deployment": agent.model_deployment,
            "status": agent.status,
            "trust_score": float(agent.trust_score),
            "reliability_score": float(agent.reliability_score),
            "average_latency_ms": int(agent.average_latency_ms),
            "hallucination_rate": float(agent.hallucination_rate),
        }

    @staticmethod
    def _mission_snapshot(mission: Mission) -> dict[str, Any]:
        return {
            "id": str(mission.id),
            "title": mission.title,
            "objective": mission.objective,
            "business_domain": mission.business_domain,
            "priority": mission.priority,
            "risk_tolerance": mission.risk_tolerance,
            "max_budget": float(mission.max_budget) if mission.max_budget is not None else None,
            "output_format": mission.output_format,
            "human_approval_preference": mission.human_approval_preference,
        }

    def _resolve_complaint_path(
        self,
        database: Session,
        *,
        mission_id: UUID,
        complaint_document_id: UUID | None,
        complaint_csv_path: str | None,
        use_default_data: bool,
        objective: str = "",
    ) -> Path:
        # `complaint_csv_path` and `use_default_data` are retained in the method
        # signature for backward compatibility with existing callers, but normal
        # mission execution is intentionally upload-only.
        _ = complaint_csv_path, use_default_data

        if complaint_document_id is None:
            raise ValueError(DATASET_REQUIRED_MESSAGE)

        document = database.get(UploadedDocument, complaint_document_id)
        if document is None or document.document_type != "complaint_csv":
            raise ValueError(DATASET_REQUIRED_MESSAGE)

        if document.mission_id is None or str(document.mission_id) != str(mission_id):
            raise ValueError(
                "Dataset mismatch — upload a source dataset specifically for this mission "
                "before execution."
            )

        if document.status != "ready":
            raise ValueError(
                "Dataset unavailable — the uploaded source dataset is not ready for execution."
            )

        path = Path(document.stored_path).resolve()
        if not path.exists() or not path.is_file() or path.suffix.lower() != ".csv":
            raise ValueError(
                "Dataset unavailable — the uploaded source CSV could not be found."
            )
        # Final server-side guard. This runs before a WorkflowRun is created, so API
        # callers cannot bypass the upload/readiness checks and accidentally spend tokens
        # on an incompatible dataset.
        ComplaintProcessor.validate_dataset_suitability(path, objective=objective)
        return path

    def data_readiness(
        self,
        database: Session,
        *,
        mission: Mission,
    ) -> dict[str, Any]:
        profile = detect_workflow_profile(mission)
        if profile == "logistics":
            readiness = LogisticsService.readiness()
            try:
                if readiness["ready"]:
                    LogisticsService.validate_data_package(readiness["data_dir"])
            except ValueError as exc:
                readiness["ready"] = False
                readiness["validation_error"] = str(exc)
            return {
                "workflow_profile": "logistics",
                **readiness,
                "validation_summary": (
                    {
                        "status": "passed",
                        "workflow_profile": "logistics",
                        "validation": "schema_and_data_quality",
                    }
                    if readiness["ready"]
                    else {}
                ),
                "message": (
                    "Logistics data package passed schema and data-quality validation."
                    if readiness["ready"]
                    else readiness.get("validation_error")
                    or "Logistics Data Required — add the three required files before execution."
                ),
            }

        document = database.scalar(
            select(UploadedDocument)
            .where(
                UploadedDocument.mission_id == mission.id,
                UploadedDocument.document_type == "complaint_csv",
                UploadedDocument.status == "ready",
            )
            .order_by(UploadedDocument.created_at.desc())
        )
        ready = False
        validation_error: str | None = None
        validation_summary: dict[str, Any] = {}
        if document is not None:
            path = Path(document.stored_path).resolve()
            ready = path.is_file() and path.suffix.lower() == ".csv"
            if ready:
                try:
                    validation_summary = ComplaintProcessor.validate_dataset_suitability(
                        path, objective=f"{mission.title} {mission.objective}"
                    )
                except ValueError as exc:
                    ready = False
                    validation_error = str(exc)
        return {
            "workflow_profile": "complaint",
            "ready": ready,
            "document_id": str(document.id) if ready and document is not None else None,
            "file_name": document.file_name if document is not None else None,
            "required_files": ["mission-linked complaint CSV"],
            "present_files": [document.file_name] if document is not None else [],
            "missing_files": [] if ready else ["valid complaint CSV"],
            "validation_error": validation_error,
            "validation_summary": validation_summary,
            "message": (
                "Complaint dataset passed schema and semantic validation."
                if ready
                else validation_error or DATASET_REQUIRED_MESSAGE
            ),
        }

    def create_run(
        self,
        database: Session,
        *,
        mission: Mission,
        actor_user_id: UUID,
        complaint_document_id: UUID | None = None,
        complaint_csv_path: str | None = None,
        operations_directory: str | None = None,
        test_mode: str = "normal",
        use_default_data: bool = False,
    ) -> WorkflowRun:
        profile = detect_workflow_profile(mission)
        requested_test_mode = str(test_mode or "normal").strip().lower()
        if profile == "complaint":
            # Continuous adversarial assurance is part of the standard complaint
            # architecture. Accept the legacy normal/adversarial request values for
            # API compatibility, but always execute the same protected complaint path.
            if requested_test_mode not in {"normal", "adversarial"}:
                raise ValueError(
                    f"Unsupported complaint workflow test mode: {requested_test_mode}."
                )
            normalized_test_mode = "adversarial"
        else:
            allowed_modes = {
                "normal",
                "logistics_reroute_failure",
                "logistics_stop_failure",
            }
            if requested_test_mode not in allowed_modes:
                raise ValueError(
                    f"Unsupported logistics workflow test mode: {requested_test_mode}."
                )
            normalized_test_mode = requested_test_mode

        complaint_path: Path | None = None
        operations_path: Path | None = None
        logistics_data_dir: Path | None = None
        if profile == "complaint":
            complaint_path = self._resolve_complaint_path(
                database,
                mission_id=mission.id,
                complaint_document_id=complaint_document_id,
                complaint_csv_path=complaint_csv_path,
                use_default_data=use_default_data,
                objective=f"{mission.title} {mission.objective}",
            )
            operations_path = Path(
                operations_directory or self.settings.default_operations_path
            ).resolve()
            if not operations_path.exists() or not operations_path.is_dir():
                raise ValueError(
                    f"Operational evidence directory is unavailable: {operations_path}"
                )
        else:
            logistics_data_dir = LogisticsService.validate_data_package()

        agents = list(
            database.scalars(
                select(Agent).where(Agent.status == "active").order_by(Agent.name)
            )
        )
        if not agents:
            raise ValueError("No active agents are registered.")

        run = self.persistence.create_run(
            database,
            mission=mission,
            initial_state={},
            actor_user_id=actor_user_id,
        )
        database.flush()
        state: dict[str, Any] = {
            "mission_id": str(mission.id),
            "workflow_run_id": str(run.id),
            "actor_user_id": str(actor_user_id),
            "mission": self._mission_snapshot(mission),
            "agent_registry": [self._agent_snapshot(agent) for agent in agents],
            "workflow_profile": profile,
            "test_mode": normalized_test_mode,
            "status": "queued",
            "current_stage": "queued",
            "progress_percent": 0.0,
            "total_tokens": 0,
            "total_latency_ms": 0,
            "estimated_cost": 0.0,
            "approval_required": False,
            "approval_id": None,
            "approval_status": None,
            "approval_comment": None,
            "retry_count": 0,
            "revision_count": 0,
            "stage_retry_counts": {},
            "failed_stage": None,
            "failed_agent_code": None,
            "original_execution_agent_code": None,
            "failed_agent_telemetry": {},
            "failure_context": {},
            "recovery_context": {},
            "skipped_stages": [],
            "recovery_required": False,
            "recovery_action": "",
            "stopped_reason": None,
            "last_error": None,
            "errors": [],
            "warnings": [],
            "timeline": [],
        }
        if complaint_path is not None:
            state["complaint_csv_path"] = str(complaint_path)
        if operations_path is not None:
            state["operations_directory"] = str(operations_path)
        if logistics_data_dir is not None:
            state["logistics_data_dir"] = str(logistics_data_dir)
        run.state_snapshot = state
        return run

    def execute_run(self, workflow_run_id: str | UUID) -> None:
        run_uuid = UUID(str(workflow_run_id))
        try:
            with SessionLocal() as database:
                run = database.get(WorkflowRun, run_uuid)
                if run is None:
                    raise RuntimeError("Workflow run not found.")
                state = dict(run.state_snapshot or {})
                self.persistence.update_run(
                    database,
                    run_id=run.id,
                    status="running",
                    current_stage="plan_mission",
                    state_snapshot={**state, "status": "running"},
                )
                database.commit()

            runtime = get_workflow_runtime()
            config = {"configurable": {"thread_id": str(run_uuid)}}
            result = runtime.graph.invoke(state, config=config)
            if result.get("__interrupt__"):
                logger.info("Workflow %s is awaiting human approval.", run_uuid)
        except Exception as exc:
            logger.exception("Workflow %s failed.", run_uuid)
            with SessionLocal() as database:
                run = database.get(WorkflowRun, run_uuid)
                if run is not None:
                    snapshot = dict(run.state_snapshot or {})
                    snapshot.update(
                        {
                            "status": "failed",
                            "last_error": str(exc),
                            "errors": list(snapshot.get("errors", [])) + [str(exc)],
                        }
                    )
                    self.persistence.update_run(
                        database,
                        run_id=run.id,
                        status="failed",
                        current_stage=str(snapshot.get("current_stage", "failed")),
                        state_snapshot=snapshot,
                        completed=True,
                    )
                    mission = database.get(Mission, run.mission_id)
                    if mission is not None:
                        mission.status = "failed"
                    database.commit()

    def resume_run(
        self,
        *,
        workflow_run_id: str | UUID,
        action: str,
        comment: str,
    ) -> None:
        run_uuid = UUID(str(workflow_run_id))
        runtime = get_workflow_runtime()
        config = {"configurable": {"thread_id": str(run_uuid)}}
        runtime.graph.invoke(
            Command(resume={"action": action, "comment": comment}),
            config=config,
        )

    @staticmethod
    def latest_run(database: Session, mission_id: UUID) -> WorkflowRun | None:
        return database.scalar(
            select(WorkflowRun)
            .where(WorkflowRun.mission_id == mission_id)
            .order_by(WorkflowRun.created_at.desc())
        )
