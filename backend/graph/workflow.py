from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from sqlalchemy import select

from backend.agents import (
    AdaptiveAgentRouter,
    AdversarialTestAgent,
    ComplaintIntelligenceAgent,
    ExecutiveReportAgent,
    GuardianGovernanceAgent,
    KnowledgeRetrievalAgent,
    LogisticsBackupExecutionAgent,
    LogisticsRoutingAgent,
    MissionPlannerAgent,
    QualityEvaluator,
    RecoverableFailureAgent,
    RecoveryController,
    RootCauseAnalysisAgent,
    TaskExecutionAgent,
    UnrecoverableFailureAgent,
    VerificationAgent,
)
from backend.config import get_settings
from backend.database import SessionLocal
from backend.graph.state import MissionState
from backend.models import Agent, Mission
from backend.services.trust_graph_service import TrustGraphService
from backend.services.workflow_persistence import WorkflowPersistenceService


class CortexWorkflowGraph:
    STAGE_PROGRESS = {
        "plan_mission": 10.0,
        "route_agents": 18.0,
        "parallel_analysis": 38.0,
        "analyse_root_causes": 55.0,
        "inject_adversarial_claim": 62.0,
        "analyse_logistics": 38.0,
        "execute_logistics": 55.0,
        "recover_logistics": 60.0,
        "reroute_logistics": 64.0,
        "execute_logistics_fallback": 67.0,
        "verify_claims": 72.0,
        "build_trust_graph": 77.0,
        "apply_governance": 84.0,
        "request_approval": 86.0,
        "evaluate_quality": 91.0,
        "generate_report": 97.0,
        "persist_final": 100.0,
    }

    def __init__(self, checkpointer: Any) -> None:
        self.settings = get_settings()
        self.persistence = WorkflowPersistenceService()
        self.planner = MissionPlannerAgent()
        self.router = AdaptiveAgentRouter()
        self.knowledge = KnowledgeRetrievalAgent()
        self.complaints = ComplaintIntelligenceAgent()
        self.root_cause = RootCauseAnalysisAgent()
        self.adversarial = AdversarialTestAgent()
        self.logistics = LogisticsRoutingAgent()
        self.task_executor = TaskExecutionAgent()
        self.logistics_backup = LogisticsBackupExecutionAgent()
        self.recoverable_failure = RecoverableFailureAgent()
        self.unrecoverable_failure = UnrecoverableFailureAgent()
        self.recovery = RecoveryController()
        self.verification = VerificationAgent()
        self.guardian = GuardianGovernanceAgent()
        self.evaluator = QualityEvaluator()
        self.reporter = ExecutiveReportAgent()
        self.graph = self._build().compile(checkpointer=checkpointer)

    @staticmethod
    def _event(
        stage: str,
        status: str,
        message: str,
        *,
        agent_code: str | None = None,
        latency_ms: int = 0,
        tokens: int = 0,
        attempt: int = 1,
    ) -> dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "status": status,
            "message": message,
            "agent_code": agent_code,
            "latency_ms": latency_ms,
            "tokens": tokens,
            "attempt": attempt,
        }

    @staticmethod
    def _result_dict(result: Any) -> dict[str, Any]:
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        return dict(result)

    def _execute_agent(
        self,
        state: MissionState,
        *,
        stage: str,
        call: Callable[[], Any],
        max_attempts: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        events: list[dict[str, Any]] = []
        last_result: dict[str, Any] = {
            "status": "failed",
            "agent_code": stage,
            "errors": ["Agent did not execute."],
        }
        maximum_attempts = (
            max(int(max_attempts), 1)
            if max_attempts is not None
            else self.settings.max_workflow_retries + 1
        )

        for attempt in range(1, maximum_attempts + 1):
            try:
                last_result = self._result_dict(call())
            except Exception as exc:
                last_result = {
                    "agent_code": stage,
                    "agent_name": stage.replace("_", " ").title(),
                    "status": "failed",
                    "output": {},
                    "confidence": 0.0,
                    "latency_ms": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost": 0.0,
                    "evidence_ids": [],
                    "warnings": [],
                    "errors": [str(exc)],
                }

            with SessionLocal() as database:
                self.persistence.record_agent_result(
                    database,
                    state=dict(state),
                    stage=stage,
                    result=last_result,
                    attempt=attempt,
                )
                success = last_result.get("status") == "completed"
                if not success and attempt < maximum_attempts:
                    run = self.persistence.update_run(
                        database,
                        run_id=state["workflow_run_id"],
                    )
                    run.retry_count = int(run.retry_count or 0) + 1
                database.commit()

            success = last_result.get("status") == "completed"
            events.append(
                self._event(
                    stage,
                    "completed" if success else "retrying",
                    (
                        f"{last_result.get('agent_name', stage)} completed."
                        if success
                        else "; ".join(last_result.get("errors", []))
                        or "Agent execution failed."
                    ),
                    agent_code=str(last_result.get("agent_code", stage)),
                    latency_ms=int(last_result.get("latency_ms", 0) or 0),
                    tokens=int(last_result.get("total_tokens", 0) or 0),
                    attempt=attempt,
                )
            )
            if success:
                return last_result, events

        events[-1]["status"] = "failed"
        return last_result, events

    def _persist_snapshot(self, state: MissionState, updates: dict[str, Any]) -> None:
        snapshot = {**dict(state), **updates}
        for key in ("total_tokens", "total_latency_ms", "estimated_cost"):
            if key in updates:
                snapshot[key] = (state.get(key, 0) or 0) + (updates.get(key, 0) or 0)
        for key in ("timeline", "warnings", "errors"):
            if key in updates:
                snapshot[key] = list(state.get(key, [])) + list(updates.get(key, []))
        with SessionLocal() as database:
            self.persistence.update_run(
                database,
                run_id=state["workflow_run_id"],
                status=str(updates.get("status", state.get("status", "running"))),
                current_stage=str(
                    updates.get("current_stage", state.get("current_stage", "running"))
                ),
                state_snapshot=snapshot,
            )
            database.commit()

    @staticmethod
    def _logistics_mission_plan(state: MissionState) -> dict[str, Any]:
        mode = str(state.get("test_mode", "normal")).lower()
        tasks: list[dict[str, Any]] = [
            {
                "code": "LOG-00",
                "title": "Define logistics mission scope and execution plan",
                "description": (
                    "Interpret the logistics objective, confirm the approved data package "
                    "and prepare the domain-specific execution plan."
                ),
                "task_type": "mission_planning",
                "stage": 1,
                "execution_mode": "sequential",
                "priority": "high",
                "risk_level": "low",
                "expected_output": "Structured logistics mission plan and execution scope.",
                "required_capabilities": ["objective understanding", "task decomposition"],
                "depends_on": [],
            },
            {
                "code": "LOG-01",
                "title": "Optimise logistics routing",
                "description": (
                    "Analyse pending orders, product weights and carrier options to produce "
                    "an evidence-backed fulfilment plan."
                ),
                "task_type": "logistics_routing",
                "stage": 2,
                "execution_mode": "sequential",
                "priority": "high",
                "risk_level": "medium",
                "expected_output": (
                    "Carrier choice, shipping cost, ETA, delivery risk and evidence catalog."
                ),
                "required_capabilities": ["carrier optimization", "logistics analysis"],
                "depends_on": ["LOG-00"],
            },
            {
                "code": "LOG-02",
                "title": "Execute simulated fulfilment task",
                "description": (
                    "Execute the approved logistics plan in simulation and produce "
                    "traceable fulfilment decisions."
                ),
                "task_type": "logistics_execution",
                "stage": 3,
                "execution_mode": "sequential",
                "priority": "high",
                "risk_level": "medium",
                "expected_output": "Simulated execution result with claim-to-evidence payload.",
                "required_capabilities": ["logistics execution", "structured output"],
                "depends_on": ["LOG-01"],
            },
        ]

        dependency = "LOG-02"
        next_stage = 4
        if mode in {"logistics_reroute_failure", "logistics_stop_failure"}:
            tasks.append(
                {
                    "code": "LOG-03",
                    "title": "Recover failed logistics execution",
                    "description": (
                        "Apply recovery policy and choose safe reroute or stop behavior "
                        "after a controlled execution failure."
                    ),
                    "task_type": "workflow_recovery",
                    "stage": next_stage,
                    "execution_mode": "conditional",
                    "priority": "critical",
                    "risk_level": "high",
                    "expected_output": (
                        "Deterministic recovery decision with reroute or stop rationale."
                    ),
                    "required_capabilities": ["workflow recovery", "failure escalation"],
                    "depends_on": ["LOG-02"],
                }
            )
            dependency = "LOG-03"
            next_stage += 1

        shared = [
            (
                "LOG-04",
                "Verify logistics claims",
                "claim_verification",
                ["claim extraction", "citation checking"],
                "Verified evidence-backed fulfilment claims.",
            ),
            (
                "LOG-05",
                "Apply logistics governance",
                "governance_review",
                ["policy enforcement", "risk classification"],
                "Governance decision and approval requirements.",
            ),
            (
                "LOG-06",
                "Evaluate mission quality",
                "quality_evaluation",
                ["quality evaluation", "performance measurement"],
                "Mission Success Score and component metrics.",
            ),
            (
                "LOG-07",
                "Generate logistics executive report",
                "report_generation",
                ["executive summarisation", "decision support"],
                "Verified logistics decision report.",
            ),
        ]
        previous = dependency
        for code, title, task_type, capabilities, expected in shared:
            tasks.append(
                {
                    "code": code,
                    "title": title,
                    "description": title + " using CORTEX shared assurance controls.",
                    "task_type": task_type,
                    "stage": next_stage,
                    "execution_mode": "sequential",
                    "priority": "high",
                    "risk_level": "medium",
                    "expected_output": expected,
                    "required_capabilities": capabilities,
                    "depends_on": [previous],
                }
            )
            previous = code
            next_stage += 1
        return {
            "mission_title": state.get("mission", {}).get("title", "Logistics Intelligence"),
            "objective": state.get("mission", {}).get("objective", ""),
            "workflow_profile": "logistics",
            "planning_method": "deterministic_domain_profile",
            "tasks": tasks,
        }


    @staticmethod
    def _ensure_complaint_assurance_plan(plan: dict[str, Any]) -> dict[str, Any]:
        """Insert the mandatory test-only adversarial assurance task.

        The business planner still creates the normal complaint tasks. CORTEX then
        inserts one deterministic assurance task immediately after root-cause
        analysis. The task is fixed to the Adversarial Test Agent and is deliberately
        excluded from adaptive agent selection.
        """
        prepared = dict(plan or {})
        raw_tasks = prepared.get("tasks", [])
        tasks = [
            dict(item)
            for item in raw_tasks
            if isinstance(item, dict)
            and str(item.get("task_type", "")) != "adversarial_testing"
        ]
        if not tasks:
            prepared["tasks"] = tasks
            return prepared

        root_index = next(
            (
                index
                for index, item in enumerate(tasks)
                if str(item.get("task_type", "")) == "root_cause_analysis"
            ),
            None,
        )
        if root_index is None:
            prepared["tasks"] = tasks
            return prepared

        root_task = tasks[root_index]
        root_code = str(root_task.get("code") or "ROOT-CAUSE")
        try:
            root_stage = int(root_task.get("stage", root_index + 1) or root_index + 1)
        except (TypeError, ValueError):
            root_stage = root_index + 1

        # Make room for the assurance stage while preserving the planner's normal
        # order and parallel groupings.
        for item in tasks:
            try:
                stage = int(item.get("stage", 0) or 0)
            except (TypeError, ValueError):
                continue
            if stage > root_stage:
                item["stage"] = stage + 1

        adversarial_code = "ADV-ASSURANCE-01"
        assurance_task = {
            "code": adversarial_code,
            "title": "Run continuous adversarial claim challenge",
            "description": (
                "Inject one deterministic unsupported RC-003 claim after normal "
                "root-cause analysis so Verification, TrustGraph and Governance "
                "must reject it before executive reporting."
            ),
            "task_type": "adversarial_testing",
            "stage": root_stage + 1,
            "execution_mode": "sequential",
            "priority": "critical",
            "risk_level": "low",
            "expected_output": (
                "RC-003 from adversarial_test_agent with TEST-FAKE-001, expected "
                "to be rejected by deterministic verification."
            ),
            "required_capabilities": [
                "controlled fault injection",
                "hallucination defense testing",
            ],
            "depends_on": [root_code],
            "fixed_agent_code": "adversarial_test_agent",
            "test_only": True,
        }
        tasks.insert(root_index + 1, assurance_task)

        # Verification must logically follow the injected challenge.
        for item in tasks:
            if str(item.get("task_type", "")) != "claim_verification":
                continue
            dependencies = [str(value) for value in item.get("depends_on", []) or []]
            if root_code in dependencies:
                dependencies = [
                    adversarial_code if value == root_code else value
                    for value in dependencies
                ]
            elif adversarial_code not in dependencies:
                dependencies.append(adversarial_code)
            item["depends_on"] = dependencies

        prepared["tasks"] = tasks
        prepared["continuous_adversarial_assurance"] = {
            "enabled": True,
            "agent_code": "adversarial_test_agent",
            "finding_id": "RC-003",
            "invalid_evidence_id": "TEST-FAKE-001",
            "expected_verification_result": "rejected",
        }
        return prepared

    @staticmethod
    def _selected_agent_code(
        state: MissionState,
        task_type: str,
        routing_key: str = "routing",
    ) -> str | None:
        routing = state.get(routing_key, []) or []
        for item in routing:
            if isinstance(item, dict) and str(item.get("task_type")) == task_type:
                value = str(item.get("selected_agent_code", "")).strip()
                return value or None
        return None

    @staticmethod
    def _agent_telemetry(agent_code: str | None) -> dict[str, Any] | None:
        code = str(agent_code or "").strip()
        if not code:
            return None
        with SessionLocal() as database:
            agent = database.scalar(select(Agent).where(Agent.code == code))
            if agent is None:
                return None
            return {
                "agent_code": agent.code,
                "agent_name": agent.name,
                "trust_score": float(agent.trust_score or 0.0),
                "reliability_score": float(agent.reliability_score or 0.0),
                "hallucination_rate": float(agent.hallucination_rate or 0.0),
                "completed_runs": int(agent.completed_runs or 0),
            }

    @staticmethod
    def _controlled_logistics_failure_context(
        mode: str,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a business-readable explanation for controlled logistics failures.

        These failure-test modes are architecture demonstrations. They are deliberately
        separated from carrier-selection business logic so users can see whether CORTEX
        reroutes or stops without corrupting normal fulfilment calculations.
        """
        errors = []
        if isinstance(result, dict):
            errors = [str(value) for value in result.get("errors", []) or [] if value]
        technical_error = "; ".join(errors)

        if mode == "logistics_reroute_failure":
            return {
                "is_controlled_test": True,
                "origin": "Controlled TEST ONLY execution-mode injection",
                "business_logic_failure": False,
                "classification": "recoverable",
                "reason": (
                    "CORTEX intentionally simulates a transient fulfilment execution timeout "
                    "after the carrier plan has already been produced successfully."
                ),
                "technical_error": technical_error or "SIMULATED_TRANSIENT_EXECUTION_TIMEOUT",
                "impact": (
                    "The orders, products, carrier data and carrier-selection result remain valid; "
                    "no simulated fulfilment action from the failed test agent is accepted."
                ),
                "recovery_policy": (
                    "Preserve the same logistics_execution task, exclude the primary and failed "
                    "test agent, and ask Adaptive Agent Router to select another compatible executor."
                ),
            }

        if mode == "logistics_stop_failure":
            return {
                "is_controlled_test": True,
                "origin": "Controlled TEST ONLY execution-mode injection",
                "business_logic_failure": False,
                "classification": "unrecoverable",
                "reason": (
                    "CORTEX intentionally simulates a non-recoverable execution-integrity fault "
                    "before fulfilment can continue safely."
                ),
                "technical_error": technical_error or "SIMULATED_EXECUTION_INTEGRITY_FAILURE",
                "impact": (
                    "The workflow cannot safely trust or continue the execution result, so no "
                    "unverified fulfilment action is allowed downstream."
                ),
                "recovery_policy": (
                    "Stop the workflow immediately; Verification, TrustGraph, Governance, Human "
                    "Approval, Quality Evaluation and Executive Report must not execute."
                ),
            }

        return {}

    @staticmethod
    def _recovery_decision_context(
        *,
        action: str,
        failure_context: dict[str, Any],
        recovery: dict[str, Any],
    ) -> dict[str, Any]:
        if action == "reroute":
            return {
                "action": "reroute",
                "why": (
                    "The failure is classified as recoverable, the evidence-backed carrier plan "
                    "is still valid, and recovery policy permits the same work to be reassigned."
                ),
                "router_instruction": (
                    "Adaptive Agent Router must exclude the original primary executor and the "
                    "failed TEST ONLY agent, then choose a different compatible logistics executor."
                ),
                "controller_reason": str(recovery.get("reason") or "Recovery policy permits rerouting."),
            }
        return {
            "action": "stop",
            "why": (
                "The failure is classified as unrecoverable. Recovery policy forbids retry or "
                "rerouting because continuing could create an unverified fulfilment result."
            ),
            "router_instruction": "No reroute is permitted for this failure class.",
            "controller_reason": str(
                recovery.get("reason")
                or failure_context.get("recovery_policy")
                or "Recovery policy stopped execution."
            ),
        }

    def _run_logistics_executor(
        self,
        agent_code: str,
        logistics_plan_result: dict[str, Any],
    ) -> Any:
        if agent_code == "task_execution":
            return self.task_executor.run_logistics(
                logistics_plan_result=logistics_plan_result
            )
        if agent_code == "logistics_backup_execution":
            return self.logistics_backup.run(
                logistics_plan_result=logistics_plan_result
            )
        raise RuntimeError(
            f"Selected logistics execution agent is not executable: {agent_code or 'unknown'}"
        )

    def plan_mission_node(self, state: MissionState) -> dict[str, Any]:
        if str(state.get("workflow_profile", "complaint")).lower() == "logistics":
            result, events = self._execute_agent(
                state,
                stage="plan_mission",
                call=lambda: self.planner.success(
                    output={"mission_plan": self._logistics_mission_plan(state)},
                    confidence=0.99,
                    latency_ms=0,
                    warnings=["Logistics mission uses the deterministic CORTEX domain plan."],
                    estimated_cost=0.0,
                ),
                max_attempts=1,
            )
        else:
            result, events = self._execute_agent(
                state,
                stage="plan_mission",
                call=lambda: self.planner.run(mission=state["mission"]),
            )
        if result.get("status") != "completed":
            updates = {
                "status": "failed",
                "current_stage": "plan_mission",
                "last_error": "; ".join(result.get("errors", [])),
                "failed_stage": "plan_mission",
                "errors": result.get("errors", []),
                "timeline": events,
            }
            self._persist_snapshot(state, updates)
            return updates

        mission_plan = result["output"]["mission_plan"]
        if str(state.get("workflow_profile", "complaint")).lower() == "complaint":
            mission_plan = self._ensure_complaint_assurance_plan(mission_plan)
            events.append(
                self._event(
                    "plan_mission",
                    "completed",
                    "Continuous adversarial assurance added as a fixed complaint-workflow stage.",
                    agent_code="mission_planner",
                )
            )

        updates = {
            "status": "running",
            "current_stage": "plan_mission",
            "progress_percent": self.STAGE_PROGRESS["plan_mission"],
            "mission_plan": mission_plan,
            "total_tokens": int(result.get("total_tokens", 0) or 0),
            "total_latency_ms": int(result.get("latency_ms", 0) or 0),
            "estimated_cost": float(result.get("estimated_cost", 0.0) or 0.0),
            "warnings": result.get("warnings", []),
            "timeline": events,
        }
        self._persist_snapshot(state, updates)
        return updates

    def route_agents_node(self, state: MissionState) -> dict[str, Any]:
        mission_tasks = list(state["mission_plan"].get("tasks", []))
        routable_tasks = [
            item
            for item in mission_tasks
            if str(item.get("task_type", "")) != "adversarial_testing"
        ]
        result, events = self._execute_agent(
            state,
            stage="route_agents",
            call=lambda: self.router.run(
                tasks=routable_tasks,
                agents=state["agent_registry"],
            ),
        )
        if result.get("status") != "completed":
            updates = {
                "status": "failed",
                "current_stage": "route_agents",
                "last_error": "; ".join(result.get("errors", [])),
                "failed_stage": "route_agents",
                "errors": result.get("errors", []),
                "timeline": events,
            }
            self._persist_snapshot(state, updates)
            return updates

        routing = list(result["output"]["routing"])
        profile = str(state.get("workflow_profile", "complaint")).lower()

        if profile == "logistics":
            # Normal logistics always starts with the designated primary executor. The
            # backup executor is deliberately reserved for recovery so the normal and
            # reroute paths are operationally and visually distinct. Adaptive Router
            # remains responsible for selecting the fallback after a recoverable failure.
            execution_task = next(
                (
                    item
                    for item in mission_tasks
                    if str(item.get("task_type", "")) == "logistics_execution"
                ),
                None,
            )
            primary_agent = next(
                (
                    item
                    for item in state.get("agent_registry", [])
                    if str(item.get("code", "")) == "task_execution"
                    and str(item.get("status", "active")) == "active"
                ),
                None,
            )
            if execution_task is not None and primary_agent is not None:
                task_code = str(execution_task.get("code", "LOG-02"))
                previous_route = next(
                    (item for item in routing if str(item.get("task_code", "")) == task_code),
                    None,
                )
                primary_route = {
                    "task_code": task_code,
                    "task_type": "logistics_execution",
                    "selected_agent_code": "task_execution",
                    "selected_agent_name": str(primary_agent.get("name") or "Task Execution Agent"),
                    "score": float((previous_route or {}).get("score", 100.0) or 100.0),
                    "reason": (
                        "Designated primary logistics executor. The Logistics Backup Execution "
                        "Agent is intentionally held in reserve and may only appear after a "
                        "Recovery Controller reroute."
                    ),
                    "alternatives": list((previous_route or {}).get("alternatives", []) or []),
                    "selection_policy": "designated_primary_with_recovery_reserve",
                }
                routing = [
                    item
                    for item in routing
                    if str(item.get("task_code", "")) != task_code
                ]
                routing.append(primary_route)
                events.append(
                    self._event(
                        "route_agents",
                        "completed",
                        "Task Execution Agent reserved as the primary logistics executor; backup execution remains recovery-only.",
                        agent_code="adaptive_agent_router",
                    )
                )

        if profile == "complaint":
            assurance_task = next(
                (
                    item
                    for item in mission_tasks
                    if str(item.get("task_type", "")) == "adversarial_testing"
                ),
                None,
            )
            if assurance_task is not None:
                routing.append(
                    {
                        "task_code": str(assurance_task.get("code", "ADV-ASSURANCE-01")),
                        "task_type": "adversarial_testing",
                        "selected_agent_code": "adversarial_test_agent",
                        "selected_agent_name": "Adversarial Test Agent",
                        "score": 100.0,
                        "reason": (
                            "Fixed continuous assurance control. This TEST ONLY agent is "
                            "not adaptively selected and cannot replace production specialists."
                        ),
                        "alternatives": [],
                        "fixed_assurance": True,
                    }
                )
                events.append(
                    self._event(
                        "route_agents",
                        "completed",
                        "Adversarial Test Agent attached as the fixed complaint assurance stage.",
                        agent_code="adaptive_agent_router",
                    )
                )

        with SessionLocal() as database:
            self.persistence.sync_plan_and_routing(
                database,
                mission_id=state["mission_id"],
                tasks=state["mission_plan"]["tasks"],
                routing=routing,
            )
            database.commit()

        updates = {
            "status": "running",
            "current_stage": "route_agents",
            "progress_percent": self.STAGE_PROGRESS["route_agents"],
            "routing": routing,
            "total_latency_ms": int(result.get("latency_ms", 0) or 0),
            "estimated_cost": float(result.get("estimated_cost", 0.0) or 0.0),
            "timeline": events,
        }
        self._persist_snapshot(state, updates)
        return updates


    def start_parallel_node(self, state: MissionState) -> dict[str, Any]:
        updates = {
            "status": "running",
            "current_stage": "parallel_analysis",
            "progress_percent": self.STAGE_PROGRESS["parallel_analysis"],
            "timeline": [
                self._event(
                    "parallel_analysis",
                    "running",
                    "Knowledge retrieval and complaint intelligence started in parallel.",
                )
            ],
        }
        self._persist_snapshot(state, updates)
        return updates

    def retrieve_knowledge_node(self, state: MissionState) -> dict[str, Any]:
        result, events = self._execute_agent(
            state,
            stage="retrieve_knowledge",
            call=lambda: self.knowledge.run(
                query=(
                    state["mission"]["objective"]
                    + " Identify applicable escalation, privacy, SLA, refund, "
                    "root-cause and human-approval policies."
                ),
                n_results=6,
            ),
        )
        return {
            "knowledge_result": result,
            "total_tokens": int(result.get("total_tokens", 0) or 0),
            "total_latency_ms": int(result.get("latency_ms", 0) or 0),
            "estimated_cost": float(result.get("estimated_cost", 0.0) or 0.0),
            "warnings": result.get("warnings", []),
            "errors": result.get("errors", []),
            "timeline": events,
        }

    def analyse_complaints_node(self, state: MissionState) -> dict[str, Any]:
        result, events = self._execute_agent(
            state,
            stage="analyse_complaints",
            call=lambda: self.complaints.run(csv_path=state["complaint_csv_path"]),
        )
        return {
            "complaint_result": result,
            "total_tokens": int(result.get("total_tokens", 0) or 0),
            "total_latency_ms": int(result.get("latency_ms", 0) or 0),
            "estimated_cost": float(result.get("estimated_cost", 0.0) or 0.0),
            "warnings": result.get("warnings", []),
            "errors": result.get("errors", []),
            "timeline": events,
        }

    def analyse_logistics_node(self, state: MissionState) -> dict[str, Any]:
        result, events = self._execute_agent(
            state,
            stage="analyse_logistics",
            call=lambda: self.logistics.run(
                data_dir=state["logistics_data_dir"],
                business_objective=state["mission"]["objective"],
                limit=5,
            ),
        )
        if result.get("status") != "completed":
            updates = {
                "status": "failed",
                "current_stage": "analyse_logistics",
                "failed_stage": "analyse_logistics",
                "last_error": "; ".join(result.get("errors", [])),
                "errors": result.get("errors", []),
                "timeline": events,
            }
            self._persist_snapshot(state, updates)
            return updates
        updates = {
            "status": "running",
            "current_stage": "analyse_logistics",
            "progress_percent": self.STAGE_PROGRESS["analyse_logistics"],
            "logistics_plan_result": result,
            "total_latency_ms": int(result.get("latency_ms", 0) or 0),
            "estimated_cost": float(result.get("estimated_cost", 0.0) or 0.0),
            "warnings": result.get("warnings", []),
            "timeline": events,
        }
        self._persist_snapshot(state, updates)
        return updates

    def execute_logistics_node(self, state: MissionState) -> dict[str, Any]:
        selected_agent = self._selected_agent_code(state, "logistics_execution")
        if not selected_agent:
            selected_agent = "task_execution"
        mode = str(state.get("test_mode", "normal")).lower()

        failure_agent_code: str | None = None
        if mode == "logistics_reroute_failure":
            failure_agent_code = self.recoverable_failure.code
            call = lambda: self.recoverable_failure.run(
                logistics_plan_result=state["logistics_plan_result"]
            )
            max_attempts = 1
        elif mode == "logistics_stop_failure":
            failure_agent_code = self.unrecoverable_failure.code
            call = lambda: self.unrecoverable_failure.run(
                logistics_plan_result=state["logistics_plan_result"]
            )
            max_attempts = 1
        else:
            call = lambda: self._run_logistics_executor(
                selected_agent,
                state["logistics_plan_result"],
            )
            max_attempts = None

        telemetry_before = self._agent_telemetry(failure_agent_code)
        result, events = self._execute_agent(
            state,
            stage="execute_logistics",
            call=call,
            max_attempts=max_attempts,
        )
        if result.get("status") != "completed":
            if mode in {"logistics_reroute_failure", "logistics_stop_failure"}:
                actual_failed_code = str(result.get("agent_code", "") or failure_agent_code or "")
                telemetry_after = self._agent_telemetry(actual_failed_code)
                failure_telemetry = None
                if telemetry_before and telemetry_after:
                    failure_telemetry = {
                        "agent_code": actual_failed_code,
                        "agent_name": telemetry_after.get("agent_name"),
                        "trust_before": telemetry_before.get("trust_score"),
                        "trust_after": telemetry_after.get("trust_score"),
                        "reliability_before": telemetry_before.get("reliability_score"),
                        "reliability_after": telemetry_after.get("reliability_score"),
                        "hallucination_rate": telemetry_after.get("hallucination_rate"),
                    }
                    events.append(
                        self._event(
                            "execute_logistics",
                            "failed",
                            (
                                f"Failure telemetry persisted for {telemetry_after.get('agent_name', actual_failed_code)}: "
                                f"trust {float(telemetry_before.get('trust_score', 0)):.2f} -> "
                                f"{float(telemetry_after.get('trust_score', 0)):.2f}; reliability "
                                f"{float(telemetry_before.get('reliability_score', 0)):.2f} -> "
                                f"{float(telemetry_after.get('reliability_score', 0)):.2f}."
                            ),
                            agent_code=actual_failed_code,
                        )
                    )
                failure_context = self._controlled_logistics_failure_context(mode, result)
                events.append(
                    self._event(
                        "execute_logistics",
                        "failed",
                        (
                            f"Controlled {failure_context.get('classification', 'test')} failure: "
                            f"{failure_context.get('reason', 'Execution failed.')} "
                            "This is not a carrier-selection business-logic failure."
                        ),
                        agent_code=actual_failed_code,
                    )
                )
                updates = {
                    "status": "running",
                    "current_stage": "execute_logistics",
                    "progress_percent": self.STAGE_PROGRESS["execute_logistics"],
                    "original_execution_agent_code": selected_agent,
                    "failed_agent_code": actual_failed_code,
                    "failed_agent_telemetry": failure_telemetry,
                    "failure_context": failure_context,
                    "failure_result": result,
                    "recovery_required": True,
                    "last_error": str(failure_context.get("reason") or "; ".join(result.get("errors", []))),
                    "warnings": result.get("warnings", []),
                    "timeline": events,
                }
                self._persist_snapshot(state, updates)
                return updates
            updates = {
                "status": "failed",
                "current_stage": "execute_logistics",
                "failed_stage": "execute_logistics",
                "failed_agent_code": str(result.get("agent_code", selected_agent)),
                "last_error": "; ".join(result.get("errors", [])),
                "errors": result.get("errors", []),
                "timeline": events,
            }
            self._persist_snapshot(state, updates)
            return updates

        assurance = result.get("output", {}).get("assurance_payload")
        if not isinstance(assurance, dict):
            updates = {
                "status": "failed",
                "current_stage": "execute_logistics",
                "failed_stage": "execute_logistics",
                "last_error": "Logistics execution did not produce an assurance payload.",
                "errors": ["Logistics assurance payload is missing."],
                "timeline": events,
            }
            self._persist_snapshot(state, updates)
            return updates

        updates = {
            "status": "running",
            "current_stage": "execute_logistics",
            "progress_percent": self.STAGE_PROGRESS["execute_logistics"],
            "original_execution_agent_code": selected_agent,
            "logistics_result": result,
            "root_cause_result": assurance,
            "recovery_required": False,
            "total_latency_ms": int(result.get("latency_ms", 0) or 0),
            "estimated_cost": float(result.get("estimated_cost", 0.0) or 0.0),
            "warnings": result.get("warnings", []),
            "timeline": events,
        }
        self._persist_snapshot(state, updates)
        return updates

    def recover_logistics_node(self, state: MissionState) -> dict[str, Any]:
        result, events = self._execute_agent(
            state,
            stage="recover_logistics",
            call=lambda: self.recovery.run(
                failed_agent_code=str(state.get("failed_agent_code") or "unknown"),
                failure_result=state.get("failure_result", {}),
                test_mode=str(state.get("test_mode", "normal")),
                available_agents=list(state.get("agent_registry", [])),
            ),
            max_attempts=1,
        )
        if result.get("status") != "completed":
            updates = {
                "status": "failed",
                "current_stage": "recover_logistics",
                "failed_stage": "workflow_stopped",
                "stopped_reason": "Recovery Controller failed to produce a safe decision.",
                "last_error": "; ".join(result.get("errors", [])),
                "errors": result.get("errors", []),
                "timeline": events,
            }
            self._persist_snapshot(state, updates)
            return updates

        recovery = result.get("output", {}).get("recovery", {})
        action = str(recovery.get("action", "stop")).lower()
        failure_context = state.get("failure_context", {}) or self._controlled_logistics_failure_context(
            str(state.get("test_mode", "normal")), state.get("failure_result", {})
        )
        recovery_context = self._recovery_decision_context(
            action=action,
            failure_context=failure_context,
            recovery=recovery if isinstance(recovery, dict) else {},
        )

        # The Recovery Controller itself succeeded when it produced a safe STOP
        # decision. Only the workflow is stopped/failed; the controller is not.
        events.append(
            self._event(
                "recover_logistics",
                "completed",
                (
                    "Recovery Controller classified the failure as recoverable and approved adaptive rerouting."
                    if action == "reroute"
                    else "Recovery Controller classified the failure as unrecoverable and issued a safe STOP decision."
                ),
                agent_code="recovery_controller",
            )
        )

        skipped_stages: list[str] = []
        if action != "reroute":
            skipped_stages = [
                "verify_claims",
                "build_trust_graph",
                "apply_governance",
                "request_approval",
                "evaluate_quality",
                "generate_report",
            ]
            for skipped_stage in skipped_stages:
                events.append(
                    self._event(
                        skipped_stage,
                        "skipped",
                        "Skipped because Recovery Controller issued STOP before downstream assurance/reporting.",
                        agent_code="recovery_controller",
                    )
                )
            with SessionLocal() as database:
                self.persistence.mark_task_types_skipped(
                    database,
                    mission_id=state["mission_id"],
                    task_types=[
                        "claim_verification",
                        "governance_review",
                        "quality_evaluation",
                        "report_generation",
                    ],
                )
                database.commit()

        stop_reason = str(
            recovery_context.get("why")
            or recovery_context.get("controller_reason")
            or "Recovery policy stopped execution."
        )
        updates = {
            "status": "running" if action == "reroute" else "failed",
            "current_stage": "recover_logistics",
            "progress_percent": self.STAGE_PROGRESS["recover_logistics"],
            "recovery_result": result,
            "recovery_context": recovery_context,
            "recovery_action": action,
            "recovery_required": False,
            "skipped_stages": skipped_stages,
            "stopped_reason": None if action == "reroute" else stop_reason,
            "failed_stage": None if action == "reroute" else "workflow_stopped",
            "last_error": None if action == "reroute" else stop_reason,
            "total_latency_ms": int(result.get("latency_ms", 0) or 0),
            "warnings": result.get("warnings", []),
            "timeline": events,
        }
        self._persist_snapshot(state, updates)
        return updates

    def reroute_logistics_node(self, state: MissionState) -> dict[str, Any]:
        execution_task = next(
            (
                task for task in state.get("mission_plan", {}).get("tasks", [])
                if isinstance(task, dict) and task.get("task_type") == "logistics_execution"
            ),
            None,
        )
        if not isinstance(execution_task, dict):
            updates = {
                "status": "failed",
                "current_stage": "reroute_logistics",
                "failed_stage": "workflow_stopped",
                "stopped_reason": "The logistics execution task could not be reconstructed for rerouting.",
                "last_error": "Recovery rerouting task is unavailable.",
                "errors": ["Recovery rerouting task is unavailable."],
                "timeline": [self._event("reroute_logistics", "failed", "Recovery rerouting task is unavailable.")],
            }
            self._persist_snapshot(state, updates)
            return updates

        excluded = {
            str(state.get("original_execution_agent_code") or ""),
            str(state.get("failed_agent_code") or ""),
        }
        candidates = [
            agent for agent in state.get("agent_registry", [])
            if str(agent.get("code")) not in excluded
        ]
        result, events = self._execute_agent(
            state,
            stage="reroute_logistics",
            call=lambda: self.router.run(tasks=[execution_task], agents=candidates),
            max_attempts=1,
        )
        if result.get("status") != "completed":
            updates = {
                "status": "failed",
                "current_stage": "reroute_logistics",
                "failed_stage": "workflow_stopped",
                "stopped_reason": "Adaptive rerouting could not find a safe compatible logistics agent.",
                "last_error": "; ".join(result.get("errors", [])),
                "errors": result.get("errors", []),
                "timeline": events,
            }
            self._persist_snapshot(state, updates)
            return updates

        routing = result.get("output", {}).get("routing", [])
        selected = str(routing[0].get("selected_agent_code", "")) if routing else ""
        if selected not in {"task_execution", "logistics_backup_execution"}:
            updates = {
                "status": "failed",
                "current_stage": "reroute_logistics",
                "failed_stage": "workflow_stopped",
                "stopped_reason": f"Reroute selected a non-executable logistics fallback: {selected or 'none'}.",
                "last_error": "No executable logistics fallback was selected.",
                "errors": ["No executable logistics fallback was selected."],
                "timeline": events,
            }
            self._persist_snapshot(state, updates)
            return updates

        with SessionLocal() as database:
            self.persistence.sync_plan_and_routing(
                database,
                mission_id=state["mission_id"],
                tasks=state["mission_plan"]["tasks"],
                routing=routing,
            )
            run = self.persistence.update_run(
                database,
                run_id=state["workflow_run_id"],
            )
            run.retry_count = int(run.retry_count or 0) + 1
            database.commit()

        events.append(
            self._event(
                "reroute_logistics",
                "completed",
                f"Adaptive Router reassigned the same logistics execution work to {routing[0].get('selected_agent_name', selected)}.",
                agent_code="adaptive_agent_router",
            )
        )
        recovery_context = dict(state.get("recovery_context", {}) or {})
        recovery_context.update(
            {
                "action": "reroute",
                "replacement_agent_code": selected,
                "replacement_agent_name": routing[0].get("selected_agent_name", selected),
                "replacement_score": float(routing[0].get("score", 0.0) or 0.0),
                "selection_reason": str(routing[0].get("reason", "")),
                "excluded_agent_codes": sorted(value for value in excluded if value),
                "final_outcome": "Replacement selected; the same logistics task continues to fallback execution.",
            }
        )
        updates = {
            "status": "running",
            "current_stage": "reroute_logistics",
            "progress_percent": self.STAGE_PROGRESS["reroute_logistics"],
            "recovery_routing": routing,
            "recovery_context": recovery_context,
            "total_latency_ms": int(result.get("latency_ms", 0) or 0),
            "timeline": events,
        }
        self._persist_snapshot(state, updates)
        return updates

    def execute_logistics_fallback_node(self, state: MissionState) -> dict[str, Any]:
        selected = self._selected_agent_code(
            state,
            "logistics_execution",
            routing_key="recovery_routing",
        )
        result, events = self._execute_agent(
            state,
            stage="execute_logistics_fallback",
            call=lambda: self._run_logistics_executor(
                str(selected or ""),
                state["logistics_plan_result"],
            ),
        )
        if result.get("status") != "completed":
            updates = {
                "status": "failed",
                "current_stage": "execute_logistics_fallback",
                "failed_stage": "workflow_stopped",
                "stopped_reason": "Fallback logistics execution also failed; CORTEX stopped safely.",
                "last_error": "; ".join(result.get("errors", [])),
                "errors": result.get("errors", []),
                "timeline": events,
            }
            self._persist_snapshot(state, updates)
            return updates

        assurance = result.get("output", {}).get("assurance_payload")
        if not isinstance(assurance, dict):
            updates = {
                "status": "failed",
                "current_stage": "execute_logistics_fallback",
                "failed_stage": "workflow_stopped",
                "stopped_reason": "Fallback agent returned no assurance payload.",
                "last_error": "Fallback assurance payload is missing.",
                "errors": ["Fallback assurance payload is missing."],
                "timeline": events,
            }
            self._persist_snapshot(state, updates)
            return updates

        recovery_context = dict(state.get("recovery_context", {}) or {})
        recovery_context["final_outcome"] = (
            "Replacement logistics execution completed successfully; CORTEX continued to "
            "Verification, TrustGraph, Governance, Quality Evaluation and Executive Report."
        )
        updates = {
            "status": "running",
            "current_stage": "execute_logistics_fallback",
            "progress_percent": self.STAGE_PROGRESS["execute_logistics_fallback"],
            "logistics_result": result,
            "root_cause_result": assurance,
            "recovery_context": recovery_context,
            "total_latency_ms": int(result.get("latency_ms", 0) or 0),
            "estimated_cost": float(result.get("estimated_cost", 0.0) or 0.0),
            "warnings": result.get("warnings", []),
            "timeline": events,
        }
        self._persist_snapshot(state, updates)
        return updates

    def analyse_root_causes_node(self, state: MissionState) -> dict[str, Any]:
        knowledge = state.get("knowledge_result", {})
        complaints = state.get("complaint_result", {})
        if knowledge.get("status") != "completed" or complaints.get("status") != "completed":
            updates = {
                "status": "failed",
                "current_stage": "parallel_analysis",
                "failed_stage": "parallel_analysis",
                "last_error": "Knowledge retrieval or complaint analysis failed.",
                "errors": ["Parallel analysis did not complete successfully."],
                "timeline": [
                    self._event(
                        "parallel_analysis",
                        "failed",
                        "Knowledge retrieval or complaint intelligence failed.",
                    )
                ],
            }
            self._persist_snapshot(state, updates)
            return updates

        knowledge_evidence = (
            knowledge.get("output", {}).get("evidence", [])
            if isinstance(knowledge.get("output"), dict)
            else []
        )
        result, events = self._execute_agent(
            state,
            stage="analyse_root_causes",
            call=lambda: self.root_cause.run(
                complaint_output=complaints,
                operations_directory=state["operations_directory"],
                knowledge_evidence=knowledge_evidence,
            ),
        )
        if result.get("status") != "completed":
            updates = {
                "status": "failed",
                "current_stage": "analyse_root_causes",
                "failed_stage": "analyse_root_causes",
                "last_error": "; ".join(result.get("errors", [])),
                "errors": result.get("errors", []),
                "timeline": events,
            }
            self._persist_snapshot(state, updates)
            return updates

        updates = {
            "status": "running",
            "current_stage": "analyse_root_causes",
            "progress_percent": self.STAGE_PROGRESS["analyse_root_causes"],
            "root_cause_result": result,
            "total_tokens": int(result.get("total_tokens", 0) or 0),
            "total_latency_ms": int(result.get("latency_ms", 0) or 0),
            "estimated_cost": float(result.get("estimated_cost", 0.0) or 0.0),
            "warnings": result.get("warnings", []),
            "timeline": events,
        }
        self._persist_snapshot(state, updates)
        return updates

    def inject_adversarial_claim_node(self, state: MissionState) -> dict[str, Any]:
        result, events = self._execute_agent(
            state,
            stage="inject_adversarial_claim",
            call=lambda: self.adversarial.run(
                root_cause_result=state["root_cause_result"]
            ),
        )
        if result.get("status") != "completed":
            updates = {
                "status": "failed",
                "current_stage": "inject_adversarial_claim",
                "failed_stage": "inject_adversarial_claim",
                "last_error": "; ".join(result.get("errors", [])),
                "errors": result.get("errors", []),
                "timeline": events,
            }
            self._persist_snapshot(state, updates)
            return updates

        output = result.get("output", {})
        modified_root_cause = output.get("root_cause_result")
        injected = output.get("injected_finding", {})
        if not isinstance(modified_root_cause, dict):
            updates = {
                "status": "failed",
                "current_stage": "inject_adversarial_claim",
                "failed_stage": "inject_adversarial_claim",
                "last_error": "Adversarial test did not return a root-cause payload.",
                "errors": ["Adversarial test output was invalid."],
                "timeline": events,
            }
            self._persist_snapshot(state, updates)
            return updates

        finding_id = str(injected.get("finding_id", "RC-003"))
        evidence_ids = injected.get("supporting_evidence_ids", [])
        submitted_evidence = str(evidence_ids[0]) if evidence_ids else "none"
        events.append(
            self._event(
                "inject_adversarial_claim",
                "completed",
                f"TEST ONLY: injected unsupported claim {finding_id} using "
                f"unapproved evidence reference {submitted_evidence}.",
                agent_code="adversarial_test_agent",
            )
        )

        adversarial_state_result = dict(result)
        adversarial_state_output = dict(output)
        adversarial_state_output.pop("root_cause_result", None)
        adversarial_state_result["output"] = adversarial_state_output

        updates = {
            "status": "running",
            "current_stage": "inject_adversarial_claim",
            "progress_percent": self.STAGE_PROGRESS["inject_adversarial_claim"],
            "root_cause_result": modified_root_cause,
            "adversarial_result": adversarial_state_result,
            "total_tokens": int(result.get("total_tokens", 0) or 0),
            "total_latency_ms": int(result.get("latency_ms", 0) or 0),
            "estimated_cost": float(result.get("estimated_cost", 0.0) or 0.0),
            "warnings": result.get("warnings", []),
            "timeline": events,
        }
        self._persist_snapshot(state, updates)
        return updates

    def verify_claims_node(self, state: MissionState) -> dict[str, Any]:
        result, events = self._execute_agent(
            state,
            stage="verify_claims",
            call=lambda: self.verification.run(
                root_cause_result=state["root_cause_result"]
            ),
        )
        if result.get("status") != "completed":
            updates = {
                "status": "failed",
                "current_stage": "verify_claims",
                "failed_stage": "verify_claims",
                "last_error": "; ".join(result.get("errors", [])),
                "errors": result.get("errors", []),
                "timeline": events,
            }
            self._persist_snapshot(state, updates)
            return updates

        with SessionLocal() as database:
            performance_feedback = self.persistence.apply_verification_feedback(
                database,
                state=dict(state),
                verification_result=result,
            )
            database.commit()

        if performance_feedback:
            events.append(
                self._event(
                    "verify_claims",
                    "completed",
                    "Agent trust telemetry updated from claim verification outcomes.",
                    agent_code="verification",
                )
            )
            for feedback in performance_feedback:
                if feedback.get("agent_code") != "adversarial_test_agent":
                    continue
                events.append(
                    self._event(
                        "verify_claims",
                        "completed",
                        (
                            "Adversarial claim feedback applied: trust "
                            f"{feedback.get('old_trust', 0):.2f} -> "
                            f"{feedback.get('new_trust', 0):.2f}; hallucination rate "
                            f"{feedback.get('old_hallucination_rate', 0):.2f}% -> "
                            f"{feedback.get('new_hallucination_rate', 0):.2f}%."
                        ),
                        agent_code="adversarial_test_agent",
                    )
                )

        updates = {
            "status": "running",
            "current_stage": "verify_claims",
            "progress_percent": self.STAGE_PROGRESS["verify_claims"],
            "verification_result": result,
            "total_tokens": int(result.get("total_tokens", 0) or 0),
            "total_latency_ms": int(result.get("latency_ms", 0) or 0),
            "estimated_cost": float(result.get("estimated_cost", 0.0) or 0.0),
            "timeline": events,
        }
        self._persist_snapshot(state, updates)
        return updates

    def build_trust_graph_node(self, state: MissionState) -> dict[str, Any]:
        graph = TrustGraphService.build(
            root_cause_result=state["root_cause_result"],
            verification_result=state["verification_result"],
            mission_id=state["mission_id"],
            workflow_profile=str(state.get("workflow_profile", "complaint")),
        ).model_dump(mode="json")
        with SessionLocal() as database:
            self.persistence.persist_trust_graph(
                database, state=dict(state), trust_graph=graph
            )
            database.commit()
        updates = {
            "current_stage": "build_trust_graph",
            "progress_percent": self.STAGE_PROGRESS["build_trust_graph"],
            "trust_graph": graph,
            "timeline": [
                self._event(
                    "build_trust_graph",
                    "completed",
                    f"Mapped {graph['summary']['claim_count']} claim(s) to "
                    f"{graph['summary']['evidence_count']} evidence item(s).",
                    agent_code="trust_graph",
                )
            ],
        }
        self._persist_snapshot(state, updates)
        return updates

    def apply_governance_node(self, state: MissionState) -> dict[str, Any]:
        result, events = self._execute_agent(
            state,
            stage="apply_governance",
            call=lambda: self.guardian.run(
                root_cause_result=state["root_cause_result"],
                verification_result=state["verification_result"],
                complaint_output=state.get("complaint_result"),
                workflow_profile=str(state.get("workflow_profile", "complaint")),
            ),
        )
        if result.get("status") != "completed":
            updates = {
                "status": "failed",
                "current_stage": "apply_governance",
                "failed_stage": "apply_governance",
                "last_error": "; ".join(result.get("errors", [])),
                "errors": result.get("errors", []),
                "timeline": events,
            }
            self._persist_snapshot(state, updates)
            return updates

        governance = result["output"]["governance"]
        with SessionLocal() as database:
            self.persistence.persist_governance(
                database, state=dict(state), governance=governance
            )
            database.commit()

        if not bool(governance.get("approval_required", False)):
            events.append(
                self._event(
                    "request_approval",
                    "skipped",
                    "Human Approval skipped because Governance determined that no high-impact action requires reviewer approval.",
                    agent_code="guardian_governance",
                )
            )
        updates = {
            "status": "running",
            "current_stage": "apply_governance",
            "progress_percent": self.STAGE_PROGRESS["apply_governance"],
            "governance_result": result,
            "approval_required": bool(governance.get("approval_required", False)),
            "total_latency_ms": int(result.get("latency_ms", 0) or 0),
            "estimated_cost": float(result.get("estimated_cost", 0.0) or 0.0),
            "warnings": result.get("warnings", []),
            "timeline": events,
        }
        self._persist_snapshot(state, updates)
        return updates

    def request_approval_node(self, state: MissionState) -> dict[str, Any]:
        governance = state["governance_result"]["output"]["governance"]
        with SessionLocal() as database:
            approval = self.persistence.ensure_approval(
                database, state=dict(state), governance=governance
            )
            self.persistence.update_run(
                database,
                run_id=state["workflow_run_id"],
                status="awaiting_approval",
                current_stage="request_approval",
                state_snapshot={
                    **dict(state),
                    "approval_id": str(approval.id),
                    "approval_status": "pending",
                },
            )
            mission = database.get(Mission, self.persistence._uuid(state["mission_id"]))
            if mission is not None:
                mission.status = "awaiting_approval"
            database.commit()

        decision = interrupt(
            {
                "approval_id": str(approval.id),
                "workflow_run_id": state["workflow_run_id"],
                "mission_id": state["mission_id"],
                "reason": approval.reason,
                "risk_level": approval.risk_level,
                "action_payload": approval.action_payload,
                "allowed_actions": ["approve", "reject", "request_revision"],
            }
        )
        if isinstance(decision, str):
            decision = {"action": decision, "comment": ""}
        action = str(decision.get("action", "reject"))
        comment = str(decision.get("comment", ""))

        governance_result = dict(state["governance_result"])
        governance_payload = dict(governance_result["output"]["governance"])
        if action == "approve":
            governance_payload["decision"] = "allow"
            governance_payload["approval_required"] = False
            governance_payload["approval_reason"] = "Approved by a human reviewer."
        elif action == "reject":
            governance_payload["decision"] = "block"
            governance_payload["approval_required"] = False
            governance_payload["approval_reason"] = "Rejected by a human reviewer."
            governance_payload["permitted_claim_ids"] = []
        else:
            revision_count = int(state.get("revision_count", 0) or 0) + 1
            if revision_count > self.settings.max_workflow_retries:
                action = "reject"
                governance_payload["decision"] = "block"
                governance_payload["approval_required"] = False
                governance_payload["approval_reason"] = (
                    "Revision limit reached; the recommendation was blocked."
                )
                governance_payload["permitted_claim_ids"] = []
            else:
                governance_payload["decision"] = "verify"
                governance_payload["approval_required"] = False
                governance_payload["approval_reason"] = "Reviewer requested revision."
        governance_result["output"] = {
            **governance_result["output"],
            "governance": governance_payload,
        }

        updates = {
            "status": "running",
            "current_stage": "request_approval",
            "progress_percent": self.STAGE_PROGRESS["request_approval"],
            "approval_id": str(approval.id),
            "approval_status": action,
            "approval_comment": comment,
            "revision_count": (
                int(state.get("revision_count", 0) or 0) + 1
                if action == "request_revision"
                else int(state.get("revision_count", 0) or 0)
            ),
            "governance_result": governance_result,
            "timeline": [
                self._event(
                    "request_approval",
                    "completed",
                    f"Human decision recorded: {action.replace('_', ' ')}.",
                    agent_code="human_reviewer",
                )
            ],
        }
        self._persist_snapshot(state, updates)
        return updates

    def evaluate_quality_node(self, state: MissionState) -> dict[str, Any]:
        result, events = self._execute_agent(
            state,
            stage="evaluate_quality",
            call=lambda: self.evaluator.run(
                root_cause_result=state["root_cause_result"],
                verification_result=state["verification_result"],
                governance_result=state["governance_result"],
                total_tokens=int(state.get("total_tokens", 0)),
                total_latency_ms=int(state.get("total_latency_ms", 0)),
            ),
        )
        if result.get("status") != "completed":
            updates = {
                "status": "failed",
                "current_stage": "evaluate_quality",
                "failed_stage": "evaluate_quality",
                "last_error": "; ".join(result.get("errors", [])),
                "errors": result.get("errors", []),
                "timeline": events,
            }
            self._persist_snapshot(state, updates)
            return updates

        evaluation = result["output"]["evaluation"]
        with SessionLocal() as database:
            self.persistence.persist_evaluation(
                database, state=dict(state), evaluation=evaluation
            )
            database.commit()
        updates = {
            "status": "running",
            "current_stage": "evaluate_quality",
            "progress_percent": self.STAGE_PROGRESS["evaluate_quality"],
            "evaluation_result": result,
            "total_latency_ms": int(result.get("latency_ms", 0) or 0),
            "estimated_cost": float(result.get("estimated_cost", 0.0) or 0.0),
            "timeline": events,
        }
        self._persist_snapshot(state, updates)
        return updates

    def generate_report_node(self, state: MissionState) -> dict[str, Any]:
        result, events = self._execute_agent(
            state,
            stage="generate_report",
            call=lambda: self.reporter.run(
                root_cause_result=state["root_cause_result"],
                verification_result=state["verification_result"],
                governance_result=state["governance_result"],
                evaluation_result=state["evaluation_result"],
                business_objective=state["mission"]["objective"],
                workflow_profile=str(state.get("workflow_profile", "complaint")),
            ),
        )
        if result.get("status") != "completed":
            updates = {
                "status": "failed",
                "current_stage": "generate_report",
                "failed_stage": "generate_report",
                "last_error": "; ".join(result.get("errors", [])),
                "errors": result.get("errors", []),
                "timeline": events,
            }
            self._persist_snapshot(state, updates)
            return updates

        updates = {
            "status": "running",
            "current_stage": "generate_report",
            "progress_percent": self.STAGE_PROGRESS["generate_report"],
            "final_report": result,
            "total_tokens": int(result.get("total_tokens", 0) or 0),
            "total_latency_ms": int(result.get("latency_ms", 0) or 0),
            "estimated_cost": float(result.get("estimated_cost", 0.0) or 0.0),
            "timeline": events,
        }
        self._persist_snapshot(state, updates)
        return updates

    def persist_final_node(self, state: MissionState) -> dict[str, Any]:
        updates = {
            "status": "completed",
            "current_stage": "completed",
            "progress_percent": 100.0,
            "timeline": [
                self._event(
                    "persist_final",
                    "completed",
                    "Mission artifacts and audit trace persisted successfully.",
                )
            ],
        }
        snapshot = {**dict(state), **updates}
        snapshot["timeline"] = list(state.get("timeline", [])) + list(updates["timeline"])
        with SessionLocal() as database:
            self.persistence.update_run(
                database,
                run_id=state["workflow_run_id"],
                status="completed",
                current_stage="completed",
                state_snapshot=snapshot,
                completed=True,
            )
            mission = database.get(Mission, self.persistence._uuid(state["mission_id"]))
            if mission is not None:
                mission.status = "completed"
            database.commit()
        return updates

    def persist_failure_node(self, state: MissionState) -> dict[str, Any]:
        updates = {
            "status": "failed",
            "current_stage": state.get("failed_stage") or "failed",
            "timeline": [
                self._event(
                    state.get("failed_stage") or "workflow",
                    "failed",
                    state.get("last_error") or "Workflow execution failed.",
                )
            ],
        }
        snapshot = {**dict(state), **updates}
        snapshot["timeline"] = list(state.get("timeline", [])) + list(updates["timeline"])
        with SessionLocal() as database:
            self.persistence.update_run(
                database,
                run_id=state["workflow_run_id"],
                status="failed",
                current_stage=updates["current_stage"],
                state_snapshot=snapshot,
                completed=True,
            )
            mission = database.get(Mission, self.persistence._uuid(state["mission_id"]))
            if mission is not None:
                mission.status = "failed"
            database.commit()
        return updates

    @staticmethod
    def route_success_or_failure(state: MissionState) -> str:
        return "failure" if state.get("status") == "failed" else "success"

    @staticmethod
    def route_after_agent_routing(state: MissionState) -> str:
        if state.get("status") == "failed":
            return "failure"
        return (
            "logistics"
            if str(state.get("workflow_profile", "complaint")).lower() == "logistics"
            else "complaint"
        )

    @staticmethod
    def route_after_root_cause(state: MissionState) -> str:
        if state.get("status") == "failed":
            return "failure"
        # Every complaint mission passes through the deterministic adversarial
        # assurance stage. The business objective never needs to request it.
        return "adversarial"

    @staticmethod
    def route_after_logistics_execution(state: MissionState) -> str:
        if state.get("status") == "failed":
            return "failure"
        if state.get("recovery_required"):
            return "recovery"
        return "success"

    @staticmethod
    def route_after_recovery(state: MissionState) -> str:
        if str(state.get("recovery_action", "stop")).lower() == "reroute" and state.get("status") != "failed":
            return "reroute"
        return "stop"

    @staticmethod
    def route_governance(state: MissionState) -> str:
        governance = (
            state.get("governance_result", {})
            .get("output", {})
            .get("governance", {})
        )
        if state.get("status") == "failed":
            return "failure"
        if governance.get("approval_required"):
            return "approval"
        return "continue"

    @staticmethod
    def route_after_approval(state: MissionState) -> str:
        if state.get("approval_status") == "request_revision":
            return "revision"
        return "continue"

    def _build(self) -> StateGraph:
        builder = StateGraph(MissionState)
        builder.add_node("plan_mission", self.plan_mission_node)
        builder.add_node("route_agents", self.route_agents_node)

        # Existing complaint workflow.
        builder.add_node("start_parallel", self.start_parallel_node)
        builder.add_node("retrieve_knowledge", self.retrieve_knowledge_node)
        builder.add_node("analyse_complaints", self.analyse_complaints_node)
        builder.add_node("analyse_root_causes", self.analyse_root_causes_node)
        builder.add_node("inject_adversarial_claim", self.inject_adversarial_claim_node)

        # Logistics workflow and recovery lane.
        builder.add_node("analyse_logistics", self.analyse_logistics_node)
        builder.add_node("execute_logistics", self.execute_logistics_node)
        builder.add_node("recover_logistics", self.recover_logistics_node)
        builder.add_node("reroute_logistics", self.reroute_logistics_node)
        builder.add_node("execute_logistics_fallback", self.execute_logistics_fallback_node)

        # Shared assurance and persistence.
        builder.add_node("verify_claims", self.verify_claims_node)
        builder.add_node("build_trust_graph", self.build_trust_graph_node)
        builder.add_node("apply_governance", self.apply_governance_node)
        builder.add_node("request_approval", self.request_approval_node)
        builder.add_node("evaluate_quality", self.evaluate_quality_node)
        builder.add_node("generate_report", self.generate_report_node)
        builder.add_node("persist_final", self.persist_final_node)
        builder.add_node("persist_failure", self.persist_failure_node)

        builder.add_edge(START, "plan_mission")
        builder.add_conditional_edges(
            "plan_mission",
            self.route_success_or_failure,
            {"success": "route_agents", "failure": "persist_failure"},
        )
        builder.add_conditional_edges(
            "route_agents",
            self.route_after_agent_routing,
            {
                "complaint": "start_parallel",
                "logistics": "analyse_logistics",
                "failure": "persist_failure",
            },
        )

        builder.add_edge("start_parallel", "retrieve_knowledge")
        builder.add_edge("start_parallel", "analyse_complaints")
        builder.add_edge(
            ["retrieve_knowledge", "analyse_complaints"],
            "analyse_root_causes",
        )
        builder.add_conditional_edges(
            "analyse_root_causes",
            self.route_after_root_cause,
            {
                "adversarial": "inject_adversarial_claim",
                "failure": "persist_failure",
            },
        )
        builder.add_conditional_edges(
            "inject_adversarial_claim",
            self.route_success_or_failure,
            {"success": "verify_claims", "failure": "persist_failure"},
        )

        builder.add_conditional_edges(
            "analyse_logistics",
            self.route_success_or_failure,
            {"success": "execute_logistics", "failure": "persist_failure"},
        )
        builder.add_conditional_edges(
            "execute_logistics",
            self.route_after_logistics_execution,
            {
                "success": "verify_claims",
                "recovery": "recover_logistics",
                "failure": "persist_failure",
            },
        )
        builder.add_conditional_edges(
            "recover_logistics",
            self.route_after_recovery,
            {"reroute": "reroute_logistics", "stop": "persist_failure"},
        )
        builder.add_conditional_edges(
            "reroute_logistics",
            self.route_success_or_failure,
            {"success": "execute_logistics_fallback", "failure": "persist_failure"},
        )
        builder.add_conditional_edges(
            "execute_logistics_fallback",
            self.route_success_or_failure,
            {"success": "verify_claims", "failure": "persist_failure"},
        )

        builder.add_conditional_edges(
            "verify_claims",
            self.route_success_or_failure,
            {"success": "build_trust_graph", "failure": "persist_failure"},
        )
        builder.add_edge("build_trust_graph", "apply_governance")
        builder.add_conditional_edges(
            "apply_governance",
            self.route_governance,
            {
                "approval": "request_approval",
                "continue": "evaluate_quality",
                "failure": "persist_failure",
            },
        )
        builder.add_conditional_edges(
            "request_approval",
            self.route_after_approval,
            {"revision": "verify_claims", "continue": "evaluate_quality"},
        )
        builder.add_conditional_edges(
            "evaluate_quality",
            self.route_success_or_failure,
            {"success": "generate_report", "failure": "persist_failure"},
        )
        builder.add_conditional_edges(
            "generate_report",
            self.route_success_or_failure,
            {"success": "persist_final", "failure": "persist_failure"},
        )
        builder.add_edge("persist_final", END)
        builder.add_edge("persist_failure", END)
        return builder
