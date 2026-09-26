from __future__ import annotations

from time import perf_counter
from typing import Any

from backend.agents.base_agent import AgentExecutionResult, BaseAgent
from backend.services.logistics_service import LogisticsService


class LogisticsBackupExecutionAgent(BaseAgent):
    code = "logistics_backup_execution"
    name = "Logistics Backup Execution Agent"

    def __init__(self) -> None:
        pass

    def run(
        self,
        *,
        logistics_plan_result: dict[str, Any],
    ) -> AgentExecutionResult:
        started_at = perf_counter()
        try:
            output = logistics_plan_result.get("output", logistics_plan_result)
            plan = output.get("logistics_plan", output)
            if not isinstance(plan, dict) or not isinstance(plan.get("shipping_plan"), list):
                raise ValueError("A valid logistics routing plan is required for fallback execution.")
            assurance = LogisticsService.build_assurance_payload(
                plan,
                source_agent_code=self.code,
            )
        except Exception as exc:
            return self.failure(
                error=f"Fallback logistics execution failed: {exc}",
                latency_ms=int((perf_counter() - started_at) * 1000),
            )

        return self.success(
            output={
                "execution": {
                    "status": "SIMULATED_COMPLETED_AFTER_REROUTE",
                    "executed_by": self.code,
                    "orders_executed": int(plan.get("orders_planned", 0) or 0),
                    "notice": "Recovery fallback completed in simulation; no external carrier booking was performed.",
                },
                "assurance_payload": assurance,
            },
            confidence=0.94,
            latency_ms=int((perf_counter() - started_at) * 1000),
            evidence_ids=[str(item["evidence_id"]) for item in plan.get("evidence_catalog", [])],
            warnings=["Recovery path used the backup logistics execution agent."],
            estimated_cost=0.0,
        )
