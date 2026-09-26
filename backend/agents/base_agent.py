from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.config import get_settings
from backend.services.llm import LLMService, TokenUsage, get_llm_service


class AgentExecutionResult(BaseModel):
    agent_code: str
    agent_name: str
    status: Literal["completed", "failed", "awaiting_approval"]
    output: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    evidence_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class BaseAgent:
    code = "base_agent"
    name = "Base Agent"

    def __init__(self, llm_service: LLMService | None = None) -> None:
        self.llm_service = llm_service or get_llm_service()
        self.settings = get_settings()

    def estimate_cost(self, usage: TokenUsage) -> float:
        input_cost = (
            usage.input_tokens
            / 1_000_000
            * self.settings.azure_openai_input_cost_per_1m
        )
        output_cost = (
            usage.output_tokens
            / 1_000_000
            * self.settings.azure_openai_output_cost_per_1m
        )
        return round(input_cost + output_cost, 8)

    def success(
        self,
        *,
        output: dict[str, Any],
        confidence: float,
        latency_ms: int,
        usage: TokenUsage | None = None,
        evidence_ids: list[str] | None = None,
        warnings: list[str] | None = None,
        estimated_cost: float | None = None,
    ) -> AgentExecutionResult:
        token_usage = usage or TokenUsage()
        cost = (
            self.estimate_cost(token_usage)
            if estimated_cost is None
            else max(float(estimated_cost), 0.0)
        )
        return AgentExecutionResult(
            agent_code=self.code,
            agent_name=self.name,
            status="completed",
            output=output,
            confidence=min(max(float(confidence), 0.0), 1.0),
            latency_ms=max(int(latency_ms), 0),
            input_tokens=token_usage.input_tokens,
            output_tokens=token_usage.output_tokens,
            total_tokens=token_usage.total_tokens,
            estimated_cost=cost,
            evidence_ids=evidence_ids or [],
            warnings=warnings or [],
        )

    def failure(
        self,
        *,
        error: str,
        latency_ms: int,
        warnings: list[str] | None = None,
    ) -> AgentExecutionResult:
        return AgentExecutionResult(
            agent_code=self.code,
            agent_name=self.name,
            status="failed",
            latency_ms=max(int(latency_ms), 0),
            errors=[str(error)],
            warnings=warnings or [],
        )
