from __future__ import annotations

from time import perf_counter
from typing import Any

from pydantic import BaseModel, Field

from backend.agents.base_agent import AgentExecutionResult, BaseAgent


class RoutingDecision(BaseModel):
    task_code: str
    task_type: str
    selected_agent_code: str
    selected_agent_name: str
    score: float = Field(ge=0.0, le=100.0)
    reason: str
    alternatives: list[dict[str, Any]] = Field(default_factory=list)


class AdaptiveAgentRouter(BaseAgent):
    code = "adaptive_agent_router"
    name = "Adaptive Agent Router"

    @staticmethod
    def _normalise(values: list[str]) -> set[str]:
        return {str(value).strip().lower() for value in values if str(value).strip()}

    @classmethod
    def score_agent(cls, task: dict[str, Any], agent: dict[str, Any]) -> float:
        if str(agent.get("status", "")).lower() != "active":
            return 0.0

        required = cls._normalise(task.get("required_capabilities", []))
        capabilities = cls._normalise(agent.get("capabilities", []))
        supported = cls._normalise(agent.get("supported_task_types", []))
        task_type = str(task.get("task_type", "")).strip().lower()

        capability_score = (
            len(required & capabilities) / max(len(required), 1) * 38.0
        )
        task_type_score = 32.0 if task_type in supported else 0.0
        trust = float(agent.get("trust_score", 0.0) or 0.0)
        reliability = float(agent.get("reliability_score", 0.0) or 0.0)
        hallucination = float(agent.get("hallucination_rate", 100.0) or 100.0)
        latency = float(agent.get("average_latency_ms", 0.0) or 0.0)

        performance_score = min(max(trust, 0.0), 100.0) * 0.14
        reliability_score = min(max(reliability, 0.0), 100.0) * 0.10
        safety_score = max(0.0, 100.0 - hallucination) * 0.04
        latency_score = max(0.0, 2.0 - min(latency / 2500.0, 2.0))

        return round(
            min(
                100.0,
                capability_score
                + task_type_score
                + performance_score
                + reliability_score
                + safety_score
                + latency_score,
            ),
            2,
        )

    def run(
        self,
        *,
        tasks: list[dict[str, Any]],
        agents: list[dict[str, Any]],
    ) -> AgentExecutionResult:
        started_at = perf_counter()
        decisions: list[RoutingDecision] = []

        try:
            for task in tasks:
                ranked = sorted(
                    (
                        (self.score_agent(task, agent), agent)
                        for agent in agents
                    ),
                    key=lambda item: item[0],
                    reverse=True,
                )
                ranked = [item for item in ranked if item[0] > 0]
                if not ranked:
                    raise RuntimeError(
                        f"No active agent can execute {task.get('task_type')}."
                    )

                best_score, best = ranked[0]
                alternatives = [
                    {
                        "agent_code": item[1].get("code"),
                        "agent_name": item[1].get("name"),
                        "score": item[0],
                    }
                    for item in ranked[1:3]
                ]
                decisions.append(
                    RoutingDecision(
                        task_code=str(task.get("code")),
                        task_type=str(task.get("task_type")),
                        selected_agent_code=str(best.get("code")),
                        selected_agent_name=str(best.get("name")),
                        score=best_score,
                        reason=(
                            "Selected for capability match, supported task type, "
                            f"trust {float(best.get('trust_score', 0)):.1f}, "
                            f"reliability {float(best.get('reliability_score', 0)):.1f}, "
                            f"and hallucination rate "
                            f"{float(best.get('hallucination_rate', 0)):.1f}%."
                        ),
                        alternatives=alternatives,
                    )
                )
        except Exception as exc:
            return self.failure(
                error=f"Agent routing failed: {exc}",
                latency_ms=int((perf_counter() - started_at) * 1000),
            )

        latency_ms = int((perf_counter() - started_at) * 1000)
        confidence = (
            sum(item.score for item in decisions) / max(len(decisions), 1) / 100.0
        )
        return self.success(
            output={"routing": [item.model_dump() for item in decisions]},
            confidence=min(confidence, 1.0),
            latency_ms=latency_ms,
        )
