from __future__ import annotations

from time import perf_counter
from typing import Any

from backend.agents.base_agent import AgentExecutionResult, BaseAgent


class RecoverableFailureAgent(BaseAgent):
    """TEST ONLY agent used to prove CORTEX recovery and adaptive rerouting."""

    code = "recoverable_failure_agent"
    name = "Recoverable Failure Test Agent"

    def __init__(self) -> None:
        pass

    def run(self, *, logistics_plan_result: dict[str, Any]) -> AgentExecutionResult:
        # The carrier plan is deliberately left untouched. Only the simulated
        # fulfilment-execution step fails, which makes the failure safely recoverable.
        _ = logistics_plan_result
        started_at = perf_counter()
        return self.failure(
            error=(
                "CONTROLLED TEST FAILURE: simulated transient fulfilment execution timeout "
                "after the evidence-backed carrier plan completed successfully. The orders, "
                "products, carrier data and routing decision are still valid; the same "
                "logistics execution work may be rerouted to another compatible agent."
            ),
            latency_ms=int((perf_counter() - started_at) * 1000),
            warnings=[
                "TEST ONLY: deterministic recoverable fault injection; not a carrier-selection or dataset failure."
            ],
        )
