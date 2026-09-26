from dataclasses import dataclass

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from backend.models import Agent, Mission, Task
from backend.services.workflow_profile import detect_workflow_profile


@dataclass(frozen=True)
class TaskBlueprint:
    title: str
    description: str
    task_type: str
    sequence_order: int
    execution_mode: str
    priority: str
    risk_level: str
    expected_output: str
    agent_code: str


def complaint_intelligence_plan() -> list[TaskBlueprint]:
    return [
        TaskBlueprint(
            title="Define mission scope and analysis plan",
            description=(
                "Interpret the complaint-intelligence objective, confirm the "
                "required outcomes and prepare the execution plan."
            ),
            task_type="mission_planning",
            sequence_order=1,
            execution_mode="sequential",
            priority="high",
            risk_level="low",
            expected_output="Structured mission plan and analysis scope.",
            agent_code="mission_planner",
        ),
        TaskBlueprint(
            title="Retrieve enterprise knowledge",
            description=(
                "Retrieve relevant support policies, SLA documents, product "
                "information and historical operational evidence."
            ),
            task_type="knowledge_retrieval",
            sequence_order=2,
            execution_mode="parallel",
            priority="high",
            risk_level="low",
            expected_output="Evidence package with source references.",
            agent_code="knowledge_retrieval",
        ),
        TaskBlueprint(
            title="Classify complaints and detect recurring patterns",
            description=(
                "Create a complaint taxonomy, classify records and identify "
                "high-volume, high-severity and emerging complaint clusters."
            ),
            task_type="complaint_analysis",
            sequence_order=2,
            execution_mode="parallel",
            priority="critical",
            risk_level="medium",
            expected_output="Complaint taxonomy, distributions and recurring patterns.",
            agent_code="complaint_intelligence",
        ),
        TaskBlueprint(
            title="Investigate root causes and business impact",
            description=(
                "Relate complaint patterns to operational evidence, classify "
                "root-cause hypotheses and estimate business impact."
            ),
            task_type="root_cause_analysis",
            sequence_order=3,
            execution_mode="sequential",
            priority="critical",
            risk_level="medium",
            expected_output=(
                "Confirmed, probable or unverified root causes with impact assessment."
            ),
            agent_code="root_cause_analysis",
        ),
        TaskBlueprint(
            title="Run continuous adversarial claim challenge",
            description=(
                "Inject one deterministic unsupported RC-003 claim after normal root-cause "
                "analysis so verification and governance must reject it before reporting."
            ),
            task_type="adversarial_testing",
            sequence_order=4,
            execution_mode="sequential",
            priority="critical",
            risk_level="low",
            expected_output=(
                "TEST ONLY RC-003 with TEST-FAKE-001, expected to be rejected by verification."
            ),
            agent_code="adversarial_test_agent",
        ),
        TaskBlueprint(
            title="Verify claims and map evidence",
            description=(
                "Check major findings against retrieved evidence, identify "
                "contradictions and calculate groundedness."
            ),
            task_type="claim_verification",
            sequence_order=5,
            execution_mode="parallel",
            priority="critical",
            risk_level="medium",
            expected_output="Claim-to-evidence map and verification results.",
            agent_code="verification",
        ),
        TaskBlueprint(
            title="Apply privacy and governance controls",
            description=(
                "Detect PII, evaluate policy compliance, classify action risk "
                "and identify recommendations requiring human approval."
            ),
            task_type="governance_review",
            sequence_order=6,
            execution_mode="parallel",
            priority="critical",
            risk_level="high",
            expected_output="Governance decision, PII status and approval requirements.",
            agent_code="guardian_governance",
        ),
        TaskBlueprint(
            title="Evaluate workflow quality",
            description=(
                "Measure task success, relevance, groundedness, hallucination "
                "risk, compliance, latency and cost efficiency."
            ),
            task_type="quality_evaluation",
            sequence_order=7,
            execution_mode="sequential",
            priority="high",
            risk_level="low",
            expected_output="Evaluation metrics and Mission Success Score components.",
            agent_code="quality_evaluator",
        ),
        TaskBlueprint(
            title="Generate verified executive report",
            description=(
                "Combine only approved and verified findings into a concise "
                "executive decision-support report."
            ),
            task_type="report_generation",
            sequence_order=8,
            execution_mode="sequential",
            priority="high",
            risk_level="medium",
            expected_output="Executive report with findings, evidence and recommendations.",
            agent_code="report_agent",
        ),
    ]


def logistics_intelligence_plan() -> list[TaskBlueprint]:
    """Create the baseline logistics plan shown immediately after mission creation.

    Recovery is deliberately not part of this baseline. It is inserted by the live
    LangGraph plan only when the user selects a recoverable or stop-on-failure run.
    """
    return [
        TaskBlueprint(
            title="Define logistics mission scope and execution plan",
            description=(
                "Interpret the logistics objective, confirm the required data package "
                "and prepare the domain-specific execution plan."
            ),
            task_type="mission_planning",
            sequence_order=1,
            execution_mode="sequential",
            priority="high",
            risk_level="low",
            expected_output="Structured logistics mission plan and execution scope.",
            agent_code="mission_planner",
        ),
        TaskBlueprint(
            title="Optimise logistics routing",
            description=(
                "Analyse pending orders, product weights and carrier options using "
                "coverage, shipping cost, ETA and carrier reliability."
            ),
            task_type="logistics_routing",
            sequence_order=2,
            execution_mode="sequential",
            priority="high",
            risk_level="medium",
            expected_output=(
                "Evidence-backed carrier selections with shipping cost, ETA and delivery risk."
            ),
            agent_code="logistics_routing",
        ),
        TaskBlueprint(
            title="Execute simulated logistics fulfilment",
            description=(
                "Execute the approved routing plan in simulation and produce traceable "
                "claim-to-evidence fulfilment decisions."
            ),
            task_type="logistics_execution",
            sequence_order=3,
            execution_mode="sequential",
            priority="high",
            risk_level="medium",
            expected_output="Simulated fulfilment execution with evidence-linked decisions.",
            agent_code="task_execution",
        ),
        TaskBlueprint(
            title="Verify logistics claims and evidence",
            description=(
                "Validate each fulfilment decision against the approved order, product "
                "and carrier evidence catalog."
            ),
            task_type="claim_verification",
            sequence_order=4,
            execution_mode="sequential",
            priority="critical",
            risk_level="medium",
            expected_output="Verified logistics claims with groundedness and citation status.",
            agent_code="verification",
        ),
        TaskBlueprint(
            title="Apply logistics governance controls",
            description=(
                "Apply risk and approval controls to evidence-backed fulfilment decisions."
            ),
            task_type="governance_review",
            sequence_order=5,
            execution_mode="sequential",
            priority="critical",
            risk_level="high",
            expected_output="Governance decision and any required human approval.",
            agent_code="guardian_governance",
        ),
        TaskBlueprint(
            title="Evaluate logistics workflow quality",
            description=(
                "Measure completion, evidence quality, safety, latency and cost efficiency."
            ),
            task_type="quality_evaluation",
            sequence_order=6,
            execution_mode="sequential",
            priority="high",
            risk_level="low",
            expected_output="Mission Success Score and logistics evaluation metrics.",
            agent_code="quality_evaluator",
        ),
        TaskBlueprint(
            title="Generate verified logistics executive report",
            description=(
                "Produce an executive fulfilment report using only verified and "
                "governance-permitted logistics findings."
            ),
            task_type="report_generation",
            sequence_order=7,
            execution_mode="sequential",
            priority="high",
            risk_level="medium",
            expected_output="Verified logistics executive report.",
            agent_code="report_agent",
        ),
    ]


def generic_enterprise_plan() -> list[TaskBlueprint]:
    return [
        TaskBlueprint(
            title="Define mission scope and execution plan",
            description="Interpret the objective and prepare a structured mission plan.",
            task_type="mission_planning",
            sequence_order=1,
            execution_mode="sequential",
            priority="high",
            risk_level="low",
            expected_output="Structured mission plan.",
            agent_code="mission_planner",
        ),
        TaskBlueprint(
            title="Retrieve relevant enterprise knowledge",
            description="Retrieve approved policies, documents and historical records.",
            task_type="knowledge_retrieval",
            sequence_order=2,
            execution_mode="parallel",
            priority="high",
            risk_level="low",
            expected_output="Evidence package with source references.",
            agent_code="knowledge_retrieval",
        ),
        TaskBlueprint(
            title="Research the business objective",
            description="Extract evidence, themes, constraints and missing information.",
            task_type="research",
            sequence_order=2,
            execution_mode="parallel",
            priority="high",
            risk_level="medium",
            expected_output="Evidence-backed research findings.",
            agent_code="research_agent",
        ),
        TaskBlueprint(
            title="Analyse findings and prepare recommendations",
            description="Identify patterns, impact, risks and practical recommendations.",
            task_type="business_analysis",
            sequence_order=3,
            execution_mode="sequential",
            priority="critical",
            risk_level="medium",
            expected_output="Business analysis and prioritised recommendations.",
            agent_code="analysis_agent",
        ),
        TaskBlueprint(
            title="Verify claims and evidence",
            description="Validate material claims and detect unsupported statements.",
            task_type="claim_verification",
            sequence_order=4,
            execution_mode="parallel",
            priority="critical",
            risk_level="medium",
            expected_output="Verification and groundedness results.",
            agent_code="verification",
        ),
        TaskBlueprint(
            title="Apply governance controls",
            description="Check privacy, policy, risk and human-approval requirements.",
            task_type="governance_review",
            sequence_order=4,
            execution_mode="parallel",
            priority="critical",
            risk_level="high",
            expected_output="Governance decision and approval requirements.",
            agent_code="guardian_governance",
        ),
        TaskBlueprint(
            title="Evaluate workflow quality",
            description="Calculate quality, groundedness, safety and efficiency metrics.",
            task_type="quality_evaluation",
            sequence_order=5,
            execution_mode="sequential",
            priority="high",
            risk_level="low",
            expected_output="Mission evaluation metrics.",
            agent_code="quality_evaluator",
        ),
        TaskBlueprint(
            title="Generate verified final report",
            description="Produce a concise report using only approved outputs.",
            task_type="report_generation",
            sequence_order=6,
            execution_mode="sequential",
            priority="high",
            risk_level="medium",
            expected_output="Verified enterprise report.",
            agent_code="report_agent",
        ),
    ]


def create_template_plan(
    database: Session,
    mission: Mission,
    replace_existing: bool = False,
) -> list[Task]:
    existing_count = database.scalar(
        select(func.count(Task.id)).where(Task.mission_id == mission.id)
    )

    if existing_count and not replace_existing:
        return list(
            database.scalars(
                select(Task)
                .where(Task.mission_id == mission.id)
                .order_by(Task.sequence_order, Task.created_at)
            ).all()
        )

    if existing_count and replace_existing:
        database.execute(delete(Task).where(Task.mission_id == mission.id))
        database.flush()

    domain = mission.business_domain.lower()
    profile = detect_workflow_profile(mission)
    if profile == "logistics":
        blueprints = logistics_intelligence_plan()
    elif "complaint" in domain:
        blueprints = complaint_intelligence_plan()
    else:
        blueprints = generic_enterprise_plan()

    required_codes = {item.agent_code for item in blueprints}

    agents = database.scalars(
        select(Agent).where(
            Agent.code.in_(required_codes),
            Agent.status.in_(["active", "test_only"]),
        )
    ).all()

    agents_by_code = {agent.code: agent for agent in agents}
    missing_codes = sorted(required_codes - set(agents_by_code))

    if missing_codes:
        raise RuntimeError(
            "Missing registered agents: " + ", ".join(missing_codes)
        )

    invalid_test_only = sorted(
        code
        for code, agent in agents_by_code.items()
        if str(agent.status).lower() == "test_only"
        and code != "adversarial_test_agent"
    )
    if invalid_test_only:
        raise RuntimeError(
            "Unexpected test-only agent in production plan: " + ", ".join(invalid_test_only)
        )

    tasks: list[Task] = []

    for item in blueprints:
        task = Task(
            mission=mission,
            assigned_agent=agents_by_code[item.agent_code],
            title=item.title,
            description=item.description,
            task_type=item.task_type,
            sequence_order=item.sequence_order,
            execution_mode=item.execution_mode,
            priority=item.priority,
            risk_level=item.risk_level,
            status="waiting",
            expected_output=item.expected_output,
        )
        database.add(task)
        tasks.append(task)

    mission.status = "planned"
    mission.plan_source = "template"
    database.flush()

    return tasks
