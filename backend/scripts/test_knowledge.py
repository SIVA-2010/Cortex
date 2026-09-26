from backend.agents.knowledge_agent import (
    KnowledgeRetrievalAgent,
)
from backend.services.vector_store import (
    get_vector_store_service,
)


def separator() -> None:
    print("-" * 68)


def main() -> None:
    separator()
    print("CORTEX KNOWLEDGE RETRIEVAL AGENT TEST")
    separator()

    vector_store = get_vector_store_service()

    health = vector_store.health_check()

    print("Chroma status :", health["status"])
    print("Collection    :", health["collection"])
    print(
        "Chunks        :",
        health["document_chunks"],
    )
    print(
        "Persist path  :",
        health["persist_path"],
    )

    if health["status"] != "ready":
        raise SystemExit(
            f"ChromaDB is not ready: {health}"
        )

    if health["document_chunks"] == 0:
        raise SystemExit(
            "The collection is empty. Run "
            "`python -m backend.scripts.ingest_knowledge` first."
        )

    separator()

    query = (
        "When must a customer complaint be escalated, "
        "and which actions need human approval?"
    )

    print("Query:", query)
    separator()

    agent = KnowledgeRetrievalAgent()

    result = agent.run(
        query=query,
        n_results=4,
    )

    print("Agent       :", result.agent_name)
    print("Status      :", result.status)
    print(
        "Confidence  :",
        f"{result.confidence:.4f}",
    )
    print("Latency ms  :", result.latency_ms)
    print("Tokens      :", result.total_tokens)
    print("Evidence    :", len(result.evidence_ids))

    if result.status != "completed":
        print("Errors:", result.errors)
        raise SystemExit(
            "Knowledge agent test failed."
        )

    separator()
    print("ANSWER")
    separator()
    print(result.output["answer"])

    separator()
    print("EVIDENCE")
    separator()

    for index, evidence in enumerate(
        result.output["evidence"],
        start=1,
    ):
        print(
            f"{index}. {evidence['title']}"
        )
        print(
            f"   ID: {evidence['evidence_id']}"
        )
        print(
            f"   Score: "
            f"{evidence['relevance_score']:.4f}"
        )
        print(
            f"   Source: {evidence['source_path']}"
        )

    separator()
    print(
        "CORTEX Knowledge Retrieval Agent "
        "completed successfully."
    )
    separator()


if __name__ == "__main__":
    main()