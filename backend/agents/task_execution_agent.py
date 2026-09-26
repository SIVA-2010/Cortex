from __future__ import annotations

from time import perf_counter
from typing import Any

from backend.agents.base_agent import AgentExecutionResult, BaseAgent
from backend.services.logistics_service import LogisticsService


class TaskExecutionAgent(BaseAgent):
    code = "task_execution"
    name = "Task Execution Agent"

    def __init__(self) -> None:
        pass

    def run_logistics(
        self,
        *,
        logistics_plan_result: dict[str, Any],
    ) -> AgentExecutionResult:
        started_at = perf_counter()
        try:
            output = logistics_plan_result.get("output", logistics_plan_result)
            plan = output.get("logistics_plan", output)
            if not isinstance(plan, dict) or not isinstance(plan.get("shipping_plan"), list):
                raise ValueError("A valid logistics routing plan is required.")
            assurance = LogisticsService.build_assurance_payload(
                plan,
                source_agent_code=self.code,
            )
        except Exception as exc:
            return self.failure(
                error=f"Logistics task execution failed: {exc}",
                latency_ms=int((perf_counter() - started_at) * 1000),
            )

        return self.success(
            output={
                "execution": {
                    "status": "SIMULATED_COMPLETED",
                    "executed_by": self.code,
                    "orders_executed": int(plan.get("orders_planned", 0) or 0),
                    "notice": "SIMULATED EXECUTION — no external carrier booking was performed.",
                },
                "assurance_payload": assurance,
            },
            confidence=0.96,
            latency_ms=int((perf_counter() - started_at) * 1000),
            evidence_ids=[str(item["evidence_id"]) for item in plan.get("evidence_catalog", [])],
            warnings=["No external carrier API was called; this is a controlled capstone simulation."],
            estimated_cost=0.0,
        )
