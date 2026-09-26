from sqlalchemy import select

from backend.database import SessionLocal, engine
from backend.models import Agent, Base, User
from backend.security import hash_password


DEMO_USERS = [
    {
        "email": "admin@cortex.com",
        "full_name": "CORTEX Administrator",
        "password": "Cortex@123",
        "role": "administrator",
    },
    {
        "email": "reviewer@cortex.com",
        "full_name": "CORTEX Reviewer",
        "password": "Review@123",
        "role": "reviewer",
    },
    {
        "email": "user@cortex.com",
        "full_name": "CORTEX Business User",
        "password": "User@123",
        "role": "business_user",
    },
]


DEMO_AGENTS = [
    {
        "code": "mission_planner",
        "name": "Mission Planner",
        "description": (
            "Understands enterprise objectives and converts them into "
            "structured, non-overlapping tasks."
        ),
        "capabilities": [
            "objective understanding",
            "task decomposition",
            "dependency planning",
            "risk classification",
        ],
        "supported_task_types": ["mission_planning"],
        "trust_score": 94.0,
        "reliability_score": 96.0,
        "average_latency_ms": 1250,
        "hallucination_rate": 2.0,
    },
    {
        "code": "knowledge_retrieval",
        "name": "Knowledge Retrieval Agent",
        "description": (
            "Retrieves approved enterprise documents, policies and "
            "historical evidence."
        ),
        "capabilities": [
            "semantic retrieval",
            "source referencing",
            "knowledge-gap detection",
        ],
        "supported_task_types": ["knowledge_retrieval"],
        "trust_score": 93.0,
        "reliability_score": 95.0,
        "average_latency_ms": 980,
        "hallucination_rate": 1.0,
    },
    {
        "code": "research_agent",
        "name": "Research Agent",
        "description": (
            "Extracts evidence, recurring themes and relevant enterprise context."
        ),
        "capabilities": [
            "evidence extraction",
            "theme discovery",
            "assumption separation",
        ],
        "supported_task_types": ["research"],
        "trust_score": 89.0,
        "reliability_score": 91.0,
        "average_latency_ms": 1450,
        "hallucination_rate": 4.0,
    },
    {
        "code": "analysis_agent",
        "name": "Enterprise Analysis Agent",
        "description": (
            "Analyses evidence, business impact, risks and practical actions."
        ),
        "capabilities": [
            "pattern analysis",
            "impact assessment",
            "recommendation planning",
        ],
        "supported_task_types": ["business_analysis"],
        "trust_score": 90.0,
        "reliability_score": 92.0,
        "average_latency_ms": 1650,
        "hallucination_rate": 4.0,
    },
    {
        "code": "complaint_intelligence",
        "name": "Complaint Intelligence Agent",
        "description": (
            "Classifies complaints and identifies recurring, severe and "
            "emerging customer issues."
        ),
        "capabilities": [
            "complaint taxonomy",
            "classification",
            "trend analysis",
            "cluster detection",
        ],
        "supported_task_types": ["complaint_analysis"],
        "trust_score": 92.0,
        "reliability_score": 94.0,
        "average_latency_ms": 1850,
        "hallucination_rate": 3.0,
    },
    {
        "code": "root_cause_analysis",
        "name": "Root-Cause Analysis Agent",
        "description": (
            "Relates business patterns to operational evidence and classifies "
            "causal findings by evidential strength."
        ),
        "capabilities": [
            "root-cause analysis",
            "causal confidence",
            "business-impact assessment",
        ],
        "supported_task_types": ["root_cause_analysis"],
        "trust_score": 91.0,
        "reliability_score": 92.0,
        "average_latency_ms": 2100,
        "hallucination_rate": 4.0,
    },
    {
        "code": "verification",
        "name": "Verification Agent",
        "description": (
            "Matches important claims to evidence and detects unsupported or "
            "contradictory statements."
        ),
        "capabilities": [
            "claim extraction",
            "citation checking",
            "groundedness scoring",
            "contradiction detection",
        ],
        "supported_task_types": ["claim_verification"],
        "trust_score": 96.0,
        "reliability_score": 97.0,
        "average_latency_ms": 1350,
        "hallucination_rate": 1.0,
    },
    {
        "code": "guardian_governance",
        "name": "Guardian Governance Agent",
        "description": (
            "Applies privacy, security, cost and risk controls and determines "
            "whether human approval is required."
        ),
        "capabilities": [
            "PII detection",
            "policy enforcement",
            "risk classification",
            "approval routing",
        ],
        "supported_task_types": ["governance_review"],
        "trust_score": 97.0,
        "reliability_score": 98.0,
        "average_latency_ms": 820,
        "hallucination_rate": 1.0,
    },
    {
        "code": "quality_evaluator",
        "name": "Quality Evaluator",
        "description": (
            "Measures relevance, groundedness, safety, reliability, latency "
            "and cost efficiency."
        ),
        "capabilities": [
            "quality evaluation",
            "hallucination scoring",
            "performance measurement",
        ],
        "supported_task_types": ["quality_evaluation"],
        "trust_score": 95.0,
        "reliability_score": 96.0,
        "average_latency_ms": 760,
        "hallucination_rate": 1.0,
    },
    {
        "code": "adaptive_agent_router",
        "name": "Adaptive Agent Router",
        "description": (
            "Selects the best agent for every task using capability, trust, "
            "reliability, latency and hallucination history."
        ),
        "capabilities": [
            "capability matching",
            "trust-based routing",
            "cost-performance routing",
        ],
        "supported_task_types": ["agent_routing"],
        "trust_score": 95.0,
        "reliability_score": 97.0,
        "average_latency_ms": 120,
        "hallucination_rate": 0.0,
    },
    {
        "code": "task_execution",
        "name": "Task Execution Agent",
        "description": (
            "Executes assigned enterprise tasks with structured outputs and "
            "standard operational metadata."
        ),
        "capabilities": [
            "task execution",
            "structured output",
            "execution metadata",
            "logistics execution",
            "simulated fulfilment",
        ],
        "supported_task_types": ["task_execution", "logistics_execution"],
        "trust_score": 89.0,
        "reliability_score": 92.0,
        "average_latency_ms": 1400,
        "hallucination_rate": 3.0,
    },
    {
        "code": "recovery_controller",
        "name": "Recovery Controller",
        "description": (
            "Retries transient failures, reroutes unsuitable work and escalates "
            "unresolved cases without duplicating actions."
        ),
        "capabilities": [
            "workflow recovery",
            "retry control",
            "failure escalation",
        ],
        "supported_task_types": ["workflow_recovery"],
        "trust_score": 96.0,
        "reliability_score": 98.0,
        "average_latency_ms": 90,
        "hallucination_rate": 0.0,
    },
    {
        "code": "adversarial_test_agent",
        "name": "Adversarial Test Agent",
        "description": (
            "Runs as the fixed TEST ONLY assurance stage in every complaint workflow, "
            "injecting one controlled unsupported claim so CORTEX continuously proves "
            "evidence rejection, hallucination detection and trust feedback."
        ),
        "capabilities": [
            "controlled fault injection",
            "hallucination defense testing",
            "verification challenge generation",
        ],
        "supported_task_types": ["adversarial_testing"],
        "model_provider": "deterministic_test",
        "model_deployment": "no_llm",
        "status": "test_only",
        "trust_score": 80.0,
        "reliability_score": 82.0,
        "average_latency_ms": 5,
        "hallucination_rate": 5.0,
    },
    {
        "code": "logistics_routing",
        "name": "Logistics Routing Agent",
        "description": (
            "Uses the approved orders, products and carrier package to build deterministic "
            "cost, ETA, coverage and reliability-aware fulfilment plans."
        ),
        "capabilities": [
            "carrier optimization",
            "logistics analysis",
            "delivery risk assessment",
            "shipping cost calculation",
        ],
        "supported_task_types": ["logistics_routing"],
        "model_provider": "deterministic_tool",
        "model_deployment": "no_llm",
        "trust_score": 92.0,
        "reliability_score": 95.0,
        "average_latency_ms": 180,
        "hallucination_rate": 0.0,
    },
    {
        "code": "logistics_backup_execution",
        "name": "Logistics Backup Execution Agent",
        "description": (
            "Provides a compatible fallback execution path for logistics work when the "
            "primary execution path cannot continue."
        ),
        "capabilities": [
            "logistics execution",
            "structured output",
            "recovery execution",
            "simulated fulfilment",
        ],
        "supported_task_types": ["logistics_execution"],
        "model_provider": "deterministic_tool",
        "model_deployment": "no_llm",
        "trust_score": 86.0,
        "reliability_score": 89.0,
        "average_latency_ms": 220,
        "hallucination_rate": 1.0,
    },
    {
        "code": "recoverable_failure_agent",
        "name": "Recoverable Failure Test Agent",
        "description": (
            "TEST ONLY agent that simulates a transient fulfilment execution timeout after valid "
            "carrier planning, so CORTEX can visibly demonstrate trust reduction, recovery "
            "classification and adaptive rerouting without changing business data."
        ),
        "capabilities": [
            "controlled fault injection",
            "recoverable failure testing",
            "logistics execution testing",
        ],
        "supported_task_types": ["logistics_execution"],
        "model_provider": "deterministic_test",
        "model_deployment": "no_llm",
        "status": "test_only",
        "trust_score": 80.0,
        "reliability_score": 82.0,
        "average_latency_ms": 5,
        "hallucination_rate": 0.0,
    },
    {
        "code": "unrecoverable_failure_agent",
        "name": "Unrecoverable Failure Test Agent",
        "description": (
            "TEST ONLY agent that simulates a non-recoverable execution-integrity failure after "
            "valid carrier planning, so CORTEX can prove safe termination and downstream "
            "stage skipping without blaming the logistics business data."
        ),
        "capabilities": [
            "controlled fault injection",
            "stop-path failure testing",
            "logistics execution testing",
        ],
        "supported_task_types": ["logistics_execution"],
        "model_provider": "deterministic_test",
        "model_deployment": "no_llm",
        "status": "test_only",
        "trust_score": 80.0,
        "reliability_score": 82.0,
        "average_latency_ms": 5,
        "hallucination_rate": 0.0,
    },
    {
        "code": "report_agent",
        "name": "Executive Report Agent",
        "description": (
            "Combines approved and verified outputs into concise enterprise reports."
        ),
        "capabilities": [
            "executive summarisation",
            "decision support",
            "explainable reporting",
        ],
        "supported_task_types": ["report_generation"],
        "trust_score": 93.0,
        "reliability_score": 95.0,
        "average_latency_ms": 1280,
        "hallucination_rate": 2.0,
    },
]


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)


def seed_users() -> None:
    with SessionLocal() as database:
        for item in DEMO_USERS:
            user = database.scalar(
                select(User).where(User.email == item["email"])
            )

            if user is None:
                user = User(email=item["email"])
                database.add(user)

            user.full_name = item["full_name"]
            user.hashed_password = hash_password(item["password"])
            user.role = item["role"]
            user.is_active = True

        database.commit()


def seed_agents() -> None:
    with SessionLocal() as database:
        for item in DEMO_AGENTS:
            agent = database.scalar(
                select(Agent).where(Agent.code == item["code"])
            )

            if agent is None:
                agent = Agent(
                    code=item["code"],
                    name=item["name"],
                    description=item["description"],
                    trust_score=item["trust_score"],
                    reliability_score=item["reliability_score"],
                    average_latency_ms=item["average_latency_ms"],
                    hallucination_rate=item["hallucination_rate"],
                    completed_runs=0,
                )
                database.add(agent)

            # Descriptive registration metadata may be refreshed safely.
            # Learned performance telemetry is intentionally preserved for
            # existing agents so rerunning init_db does not reset AgentOps history.
            agent.name = item["name"]
            agent.description = item["description"]
            agent.capabilities = item["capabilities"]
            agent.supported_task_types = item["supported_task_types"]
            agent.model_provider = item.get("model_provider", "azure_openai")
            agent.model_deployment = item.get("model_deployment", "cortex-llm")
            agent.status = item.get("status", "active")

        database.commit()


def main() -> None:
    create_tables()
    seed_users()
    seed_agents()

    print("CORTEX tables and seed data are ready.")
    print(f"Seeded users: {len(DEMO_USERS)}")
    print(f"Seeded agents: {len(DEMO_AGENTS)}")


if __name__ == "__main__":
    main()
