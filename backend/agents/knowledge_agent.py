from __future__ import annotations

from statistics import mean
from time import perf_counter

from backend.agents.base_agent import (
    AgentExecutionResult,
    BaseAgent,
)
from backend.services.vector_store import (
    VectorStoreService,
    get_vector_store_service,
)


class KnowledgeRetrievalAgent(BaseAgent):
    code = "knowledge_retrieval"
    name = "Knowledge Retrieval Agent"

    def __init__(
        self,
        vector_store: VectorStoreService | None = None,
    ) -> None:
        super().__init__()

        self.vector_store = (
            vector_store
            or get_vector_store_service()
        )

    def run(
        self,
        *,
        query: str,
        n_results: int = 5,
    ) -> AgentExecutionResult:
        started_at = perf_counter()

        try:
            evidence = self.vector_store.search(
                query,
                n_results=n_results,
            )

        except Exception as exc:
            latency_ms = int(
                (perf_counter() - started_at) * 1000
            )

            return self.failure(
                error=(
                    "Enterprise knowledge retrieval failed: "
                    f"{exc}"
                ),
                latency_ms=latency_ms,
            )

        if not evidence:
            latency_ms = int(
                (perf_counter() - started_at) * 1000
            )

            return self.success(
                output={
                    "query": query,
                    "answer": (
                        "No approved enterprise evidence was "
                        "available for this query."
                    ),
                    "evidence": [],
                    "missing_information": True,
                },
                confidence=0.0,
                latency_ms=latency_ms,
                warnings=[
                    "The enterprise knowledge collection is empty "
                    "or no relevant evidence was found."
                ],
            )

        context_blocks: list[str] = []

        for index, item in enumerate(
            evidence,
            start=1,
        ):
            context_blocks.append(
                "\n".join(
                    [
                        f"[E{index}]",
                        f"Title: {item.title}",
                        f"Source: {item.source_path}",
                        (
                            "Relevance score: "
                            f"{item.relevance_score:.4f}"
                        ),
                        f"Evidence text: {item.text}",
                    ]
                )
            )

        evidence_context = "\n\n".join(
            context_blocks
        )

        system_prompt = """
        You are the CORTEX Knowledge Retrieval Agent.

        Answer the business question using only the approved
        enterprise evidence supplied in the prompt.

        Requirements:

        1. Do not use unsupported external knowledge.
        2. Cite evidence using labels such as [E1] and [E2].
        3. Clearly state when evidence is incomplete.
        4. Separate policy facts from recommendations.
        5. Keep the answer concise and enterprise-friendly.
        """

        user_prompt = f"""
        Business question:

        {query}

        Approved enterprise evidence:

        {evidence_context}

        Produce an evidence-grounded answer with citations.
        """

        first_evidence = evidence[0]

        mock_text = (
            f"Based on [{first_evidence.evidence_id}], "
            f"{first_evidence.title} states: "
            f"{first_evidence.text[:500]}"
        )

        llm_result = self.llm_service.invoke_text(
            agent_name=self.name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            mock_text=mock_text,
        )

        total_latency_ms = int(
            (perf_counter() - started_at) * 1000
        )

        if llm_result.status != "completed":
            return self.failure(
                error=(
                    "Knowledge answer generation failed: "
                    f"{llm_result.error}"
                ),
                latency_ms=total_latency_ms,
            )

        average_relevance = mean(
            item.relevance_score
            for item in evidence
        )

        confidence = min(
            max(average_relevance, 0.0),
            1.0,
        )

        return self.success(
            output={
                "query": query,
                "answer": str(llm_result.content),
                "evidence": [
                    item.model_dump()
                    for item in evidence
                ],
                "missing_information": False,
                "provider": llm_result.provider,
                "deployment": llm_result.deployment,
            },
            confidence=confidence,
            latency_ms=total_latency_ms,
            usage=llm_result.usage,
            evidence_ids=[
                item.evidence_id
                for item in evidence
            ],
        )