from __future__ import annotations

from time import perf_counter

from backend.agents.base_agent import AgentExecutionResult, BaseAgent
from backend.services.logistics_service import LogisticsService


class LogisticsRoutingAgent(BaseAgent):
    code = "logistics_routing"
    name = "Logistics Routing Agent"

    def __init__(self) -> None:
        pass

    def run(
        self,
        *,
        data_dir: str,
        business_objective: str,
        limit: int = 5,
    ) -> AgentExecutionResult:
        started_at = perf_counter()
        try:
            plan = LogisticsService.build_shipping_plan(
                data_dir=data_dir,
                objective=business_objective,
                limit=limit,
            )
        except Exception as exc:
            return self.failure(
                error=f"Logistics routing failed: {exc}",
                latency_ms=int((perf_counter() - started_at) * 1000),
            )

        confidence = 0.95 if plan.get("orders_planned", 0) else 0.55
        warnings = [
            "SIMULATED EXECUTION: carrier selections are planning outputs only; no booking is performed."
        ]
        if plan.get("exceptions"):
            warnings.append(f"{len(plan['exceptions'])} logistics exception(s) require review.")
        return self.success(
            output={"logistics_plan": plan},
            confidence=confidence,
            latency_ms=int((perf_counter() - started_at) * 1000),
            evidence_ids=[str(item["evidence_id"]) for item in plan.get("evidence_catalog", [])],
            warnings=warnings,
            estimated_cost=0.0,
        )
