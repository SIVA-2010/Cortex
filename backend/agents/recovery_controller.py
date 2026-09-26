from __future__ import annotations

from time import perf_counter
from typing import Any

from backend.agents.base_agent import AgentExecutionResult, BaseAgent


class RecoveryController(BaseAgent):
    code = "recovery_controller"
    name = "Recovery Controller"

    def __init__(self) -> None:
        pass

    def run(
        self,
        *,
        failed_agent_code: str,
        failure_result: dict[str, Any],
        test_mode: str,
        available_agents: list[dict[str, Any]],
    ) -> AgentExecutionResult:
        started_at = perf_counter()
        mode = str(test_mode or "normal").lower()
        errors = failure_result.get("errors", []) if isinstance(failure_result, dict) else []
        failure_reason = "; ".join(str(value) for value in errors) or "Execution failed."

        compatible = [
            agent
            for agent in available_agents
            if str(agent.get("status", "")).lower() == "active"
            and "logistics_execution" in {
                str(value).strip().lower()
                for value in agent.get("supported_task_types", [])
            }
            and str(agent.get("code")) != str(failed_agent_code)
        ]

        if mode == "logistics_reroute_failure" and compatible:
            action = "reroute"
            classification = "recoverable"
            reason = (
                "The controlled execution failure is recoverable: the carrier plan remains valid "
                f"and {len(compatible)} active compatible logistics executor(s) are available. "
                "CORTEX therefore preserves the same work and asks Adaptive Agent Router to select "
                "a different executor."
            )
        else:
            action = "stop"
            classification = "unrecoverable"
            reason = (
                "The controlled execution failure is non-recoverable for this policy path, or no "
                "safe compatible executor is available. CORTEX therefore stops before downstream "
                "verification and reporting rather than manufacture an unverified result."
            )

        return self.success(
            output={
                "recovery": {
                    "action": action,
                    "classification": classification,
                    "failed_agent_code": str(failed_agent_code),
                    "failure_reason": failure_reason,
                    "compatible_fallback_agents": [
                        str(agent.get("code")) for agent in compatible
                    ],
                    "reason": reason,
                    "test_only": mode in {
                        "logistics_reroute_failure",
                        "logistics_stop_failure",
                    },
                }
            },
            confidence=1.0,
            latency_ms=int((perf_counter() - started_at) * 1000),
            warnings=["Recovery decision is deterministic and policy-controlled."],
            estimated_cost=0.0,
        )
