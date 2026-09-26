from __future__ import annotations

import json
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from backend.agents.base_agent import AgentExecutionResult, BaseAgent
from backend.services.mission_planner import (
    complaint_intelligence_plan,
    generic_enterprise_plan,
)


ExecutionMode = Literal["sequential", "parallel", "conditional"]
Priority = Literal["low", "medium", "high", "critical"]
RiskLevel = Literal["low", "medium", "high", "prohibited"]


class PlannedTask(BaseModel):
    code: str = Field(min_length=2, max_length=80)
    title: str = Field(min_length=3, max_length=200)
    description: str = Field(min_length=10, max_length=1500)
    task_type: str = Field(min_length=2, max_length=80)
    stage: int = Field(ge=1, le=20)
    execution_mode: ExecutionMode = "sequential"
    priority: Priority = "medium"
    risk_level: RiskLevel = "low"
    expected_output: str = Field(min_length=5, max_length=1000)
    required_capabilities: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class MissionPlanOutput(BaseModel):
    mission_summary: str
    expected_outcomes: list[str]
    tasks: list[PlannedTask]
    assumptions: list[str] = Field(default_factory=list)
    data_requirements: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_task_codes(self) -> MissionPlanOutput:
        codes = [task.code for task in self.tasks]
        if len(codes) != len(set(codes)):
            raise ValueError("Task codes must be unique.")
        known = set(codes)
        for task in self.tasks:
            missing = sorted(set(task.depends_on) - known)
            if missing:
                raise ValueError(
                    f"Task {task.code} has unknown dependencies: {missing}"
                )
        return self


TASK_CAPABILITIES: dict[str, list[str]] = {
    "mission_planning": ["task decomposition", "dependency planning"],
    "knowledge_retrieval": ["semantic retrieval", "source referencing"],
    "complaint_analysis": ["classification", "trend analysis", "cluster detection"],
    "root_cause_analysis": ["root-cause analysis", "business-impact assessment"],
    "claim_verification": ["claim extraction", "groundedness scoring"],
    "governance_review": ["PII detection", "policy enforcement"],
    "quality_evaluation": ["quality evaluation", "hallucination scoring"],
    "report_generation": ["executive summarisation", "decision support"],
    "research": ["evidence extraction", "theme discovery"],
    "business_analysis": ["pattern analysis", "impact assessment"],
}


class MissionPlannerAgent(BaseAgent):
    code = "mission_planner"
    name = "Mission Planner"

    @staticmethod
    def _baseline(mission: dict[str, Any]) -> MissionPlanOutput:
        domain = str(mission.get("business_domain", "")).lower()
        blueprints = (
            complaint_intelligence_plan()
            if "complaint" in domain
            else generic_enterprise_plan()
        )

        task_codes: dict[int, list[str]] = {}
        tasks: list[PlannedTask] = []
        for index, item in enumerate(blueprints, start=1):
            code = f"{item.task_type}-{index:02d}"
            prior_codes = [
                candidate
                for stage, candidates in task_codes.items()
                if stage < item.sequence_order
                for candidate in candidates
            ]
            depends_on = prior_codes[-1:] if item.sequence_order > 1 else []
            task_codes.setdefault(item.sequence_order, []).append(code)
            tasks.append(
                PlannedTask(
                    code=code,
                    title=item.title,
                    description=item.description,
                    task_type=item.task_type,
                    stage=item.sequence_order,
                    execution_mode=item.execution_mode,
                    priority=item.priority,
                    risk_level=(
                        "high" if item.risk_level == "critical" else item.risk_level
                    ),
                    expected_output=item.expected_output,
                    required_capabilities=TASK_CAPABILITIES.get(
                        item.task_type, [item.task_type.replace("_", " ")]
                    ),
                    depends_on=depends_on,
                )
            )

        return MissionPlanOutput(
            mission_summary=(
                f"Execute {mission.get('title', 'the enterprise mission')} using "
                "a governed, evidence-grounded multi-agent workflow."
            ),
            expected_outcomes=[
                "A complete and non-overlapping execution plan.",
                "Evidence-grounded findings with transparent agent selection.",
                "A governance decision and an executive-ready final report.",
            ],
            tasks=tasks,
            assumptions=[
                "The selected enterprise data sources are approved for this mission.",
                "All demonstration records are synthetic unless explicitly stated otherwise.",
            ],
            data_requirements=[
                "Complaint CSV or another structured mission dataset.",
                "Approved enterprise policies in the CORTEX knowledge base.",
                "Operational evidence required for causal verification.",
            ],
        )

    @staticmethod
    def _guard(candidate: MissionPlanOutput, baseline: MissionPlanOutput) -> MissionPlanOutput:
        mandatory = {task.task_type for task in baseline.tasks}
        supplied = {task.task_type for task in candidate.tasks}
        if not mandatory.issubset(supplied):
            return baseline

        candidate.tasks = sorted(candidate.tasks, key=lambda item: (item.stage, item.code))
        for task in candidate.tasks:
            task.required_capabilities = task.required_capabilities or TASK_CAPABILITIES.get(
                task.task_type, [task.task_type.replace("_", " ")]
            )
        return candidate

    def run(self, *, mission: dict[str, Any]) -> AgentExecutionResult:
        started_at = perf_counter()
        baseline = self._baseline(mission)

        system_prompt = """
        You are the CORTEX Mission Planner.

        Convert the supplied enterprise objective into a complete, concise and
        non-overlapping workflow. Preserve the required structured schema.

        Rules:
        - Include planning, retrieval, domain analysis, verification, governance,
          evaluation and reporting.
        - Use parallel execution only when tasks are genuinely independent.
        - Mark high-impact or sensitive work as high risk.
        - Dependencies must reference valid task codes.
        - Do not invent unavailable data sources.
        - Keep task descriptions practical and demo-ready.
        """
        user_prompt = (
            "Create the mission plan for this request:\n"
            + json.dumps(mission, ensure_ascii=False, indent=2, default=str)
            + "\n\nSafe baseline:\n"
            + baseline.model_dump_json(indent=2)
        )

        result = self.llm_service.invoke_structured(
            agent_name=self.name,
            schema=MissionPlanOutput,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            mock_value=baseline,
        )
        latency_ms = int((perf_counter() - started_at) * 1000)
        if result.status != "completed":
            return self.failure(
                error=f"Mission planning failed: {result.error}",
                latency_ms=latency_ms,
            )

        output = result.content
        if not isinstance(output, MissionPlanOutput):
            try:
                output = MissionPlanOutput.model_validate(output)
            except Exception as exc:
                return self.failure(
                    error=f"Invalid mission plan: {exc}", latency_ms=latency_ms
                )
        output = self._guard(output, baseline)

        return self.success(
            output={
                "mission_plan": output.model_dump(),
                "provider": result.provider,
                "deployment": result.deployment,
            },
            confidence=0.95,
            latency_ms=latency_ms,
            usage=result.usage,
        )
