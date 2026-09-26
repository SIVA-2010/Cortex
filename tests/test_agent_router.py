from backend.agents.agent_router import AdaptiveAgentRouter


def test_router_prefers_supported_trusted_agent() -> None:
    task = {
        "task_type": "claim_verification",
        "required_capabilities": ["claim extraction", "groundedness scoring"],
    }
    strong = {
        "status": "active",
        "capabilities": ["claim extraction", "groundedness scoring"],
        "supported_task_types": ["claim_verification"],
        "trust_score": 96,
        "reliability_score": 97,
        "hallucination_rate": 1,
        "average_latency_ms": 1200,
    }
    weak = {
        "status": "active",
        "capabilities": ["summarisation"],
        "supported_task_types": ["report_generation"],
        "trust_score": 80,
        "reliability_score": 80,
        "hallucination_rate": 10,
        "average_latency_ms": 2000,
    }
    assert AdaptiveAgentRouter.score_agent(task, strong) > AdaptiveAgentRouter.score_agent(task, weak)
