from __future__ import annotations

from time import perf_counter
from typing import Any

from backend.agents.base_agent import AgentExecutionResult, BaseAgent


class UnrecoverableFailureAgent(BaseAgent):
    """TEST ONLY agent used to prove safe termination of an unsafe execution path."""

    code = "unrecoverable_failure_agent"
    name = "Unrecoverable Failure Test Agent"

    def __init__(self) -> None:
        pass

    def run(self, *, logistics_plan_result: dict[str, Any]) -> AgentExecutionResult:
        # The deterministic carrier plan itself is not corrupted. This controlled
        # failure represents an execution-integrity condition for which policy forbids
        # retry or reroute, so no unverified fulfilment action can proceed downstream.
        _ = logistics_plan_result
        started_at = perf_counter()
        return self.failure(
            error=(
                "CONTROLLED TEST FAILURE: simulated non-recoverable execution-integrity fault "
                "after the evidence-backed carrier plan completed successfully. Continuing or "
                "rerouting this execution result is not permitted, so CORTEX must stop before "
                "Verification, Governance, Quality Evaluation or reporting."
            ),
            latency_ms=int((perf_counter() - started_at) * 1000),
            warnings=[
                "TEST ONLY: deterministic stop-path fault injection; not a carrier-selection or dataset failure."
            ],
        )
