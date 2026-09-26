from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TrustGraphNode(BaseModel):
    node_id: str
    node_type: str
    label: str
    status: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrustGraphEdge(BaseModel):
    source: str
    target: str
    relationship: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrustGraphOutput(BaseModel):
    mission_id: str
    nodes: list[TrustGraphNode]
    edges: list[TrustGraphEdge]
    claim_evidence_table: list[dict[str, Any]]
    summary: dict[str, Any]


class TrustGraphService:
    @staticmethod
    def _root_payload(root_cause_result: dict[str, Any]) -> dict[str, Any]:
        payload = root_cause_result
        if isinstance(payload.get("output"), dict):
            payload = payload["output"]
        return payload

    @staticmethod
    def _verification_payload(
        verification_result: dict[str, Any],
    ) -> dict[str, Any]:
        payload = verification_result
        if isinstance(payload.get("output"), dict):
            payload = payload["output"]
        if isinstance(payload.get("verification"), dict):
            payload = payload["verification"]
        return payload

    @classmethod
    def build(
        cls,
        *,
        root_cause_result: dict[str, Any],
        verification_result: dict[str, Any],
        mission_id: str = "DEMO-MISSION-001",
        workflow_profile: str = "complaint",
    ) -> TrustGraphOutput:
        root_payload = cls._root_payload(root_cause_result)
        verification = cls._verification_payload(verification_result)

        operational = (
            root_payload.get("operational_evidence_summary")
            or root_payload.get("operational_summary")
            or {}
        )
        catalog = operational.get("evidence_catalog", [])
        if not isinstance(catalog, list):
            catalog = []
        evidence_by_id = {
            str(item.get("evidence_id")): item
            for item in catalog
            if isinstance(item, dict) and item.get("evidence_id")
        }

        claims = verification.get("claims", [])
        if not isinstance(claims, list):
            claims = []

        profile = str(workflow_profile or "complaint").lower()
        nodes: list[TrustGraphNode] = [
            TrustGraphNode(
                node_id=mission_id,
                node_type="mission",
                label=(
                    "Logistics Intelligence Mission"
                    if profile == "logistics"
                    else "Customer Complaint Intelligence Mission"
                ),
                status="executed",
                metadata={"synthetic_data": True, "workflow_profile": profile},
            ),
            TrustGraphNode(
                node_id="AGENT-verification",
                node_type="agent",
                label="Verification Agent",
                status="completed",
            ),
        ]
        edges: list[TrustGraphEdge] = []
        table: list[dict[str, Any]] = []
        added_evidence: set[str] = set()
        added_invalid_references: set[str] = set()
        added_agent_nodes: set[str] = {"verification"}
        agent_labels = {
            "root_cause_analysis": "Root-Cause Analysis Agent",
            "adversarial_test_agent": "Adversarial Test Agent - TEST ONLY",
            "task_execution": "Task Execution Agent",
            "logistics_backup_execution": "Logistics Backup Execution Agent",
        }

        for claim in claims:
            if not isinstance(claim, dict):
                continue
            claim_id = str(claim.get("claim_id", ""))
            if not claim_id:
                continue

            source_agent_code = str(
                claim.get("source_agent_code") or "root_cause_analysis"
            ).strip() or "root_cause_analysis"
            source_agent_node = f"AGENT-{source_agent_code}"
            if source_agent_code not in added_agent_nodes:
                nodes.append(
                    TrustGraphNode(
                        node_id=source_agent_node,
                        node_type="agent",
                        label=agent_labels.get(
                            source_agent_code,
                            source_agent_code.replace("_", " ").title(),
                        ),
                        status=(
                            "test_only"
                            if source_agent_code == "adversarial_test_agent"
                            else "completed"
                        ),
                        metadata={
                            "agent_code": source_agent_code,
                            "test_only": source_agent_code == "adversarial_test_agent",
                        },
                    )
                )
                added_agent_nodes.add(source_agent_code)

            nodes.append(
                TrustGraphNode(
                    node_id=claim_id,
                    node_type="claim",
                    label=str(claim.get("claim", "Root-cause claim")),
                    status=str(claim.get("verification_result", "unknown")),
                    metadata={
                        "finding_id": claim.get("finding_id"),
                        "source_agent_code": source_agent_code,
                        "category": claim.get("category"),
                        "groundedness": claim.get("groundedness"),
                        "hallucination_risk": claim.get("hallucination_risk"),
                        "citation_status": claim.get("citation_status"),
                    },
                )
            )
            edges.extend(
                [
                    TrustGraphEdge(
                        source=mission_id,
                        target=claim_id,
                        relationship="contains_claim",
                    ),
                    TrustGraphEdge(
                        source=source_agent_node,
                        target=claim_id,
                        relationship="produced",
                    ),
                    TrustGraphEdge(
                        source="AGENT-verification",
                        target=claim_id,
                        relationship="verified",
                        metadata={
                            "verification_result": claim.get("verification_result")
                        },
                    ),
                ]
            )

            evidence_ids = claim.get("evidence_ids", [])
            if not isinstance(evidence_ids, list):
                evidence_ids = []

            for evidence_id_value in evidence_ids:
                evidence_id = str(evidence_id_value)
                evidence = evidence_by_id.get(evidence_id, {})
                if evidence_id not in added_evidence:
                    nodes.append(
                        TrustGraphNode(
                            node_id=evidence_id,
                            node_type="evidence",
                            label=str(
                                evidence.get("description")
                                or evidence.get("summary")
                                or evidence.get("change_summary")
                                or evidence.get("category")
                                or evidence_id
                            )[:180],
                            status=str(evidence.get("strength", "supporting")),
                            metadata={
                                "evidence_type": evidence.get("evidence_type"),
                                "category": evidence.get("category"),
                                "observed_at": evidence.get("observed_at"),
                            },
                        )
                    )
                    added_evidence.add(evidence_id)

                edges.append(
                    TrustGraphEdge(
                        source=claim_id,
                        target=evidence_id,
                        relationship="supported_by",
                    )
                )

            invalid_evidence_ids = claim.get("invalid_evidence_ids", [])
            if not isinstance(invalid_evidence_ids, list):
                invalid_evidence_ids = []

            for invalid_id_value in invalid_evidence_ids:
                invalid_id = str(invalid_id_value)
                invalid_node_id = f"INVALID-{invalid_id}"
                if invalid_id not in added_invalid_references:
                    nodes.append(
                        TrustGraphNode(
                            node_id=invalid_node_id,
                            node_type="invalid_evidence",
                            label=f"Rejected reference: {invalid_id}",
                            status="invalid",
                            metadata={
                                "evidence_id": invalid_id,
                                "approved": False,
                                "reason": "Reference not found in approved evidence catalog.",
                            },
                        )
                    )
                    added_invalid_references.add(invalid_id)
                edges.append(
                    TrustGraphEdge(
                        source=claim_id,
                        target=invalid_node_id,
                        relationship="rejected_reference",
                        metadata={"approved": False},
                    )
                )

            table.append(
                {
                    "claim_id": claim_id,
                    "finding_id": claim.get("finding_id"),
                    "source_agent_code": source_agent_code,
                    "category": claim.get("category"),
                    "claim": claim.get("claim"),
                    "verification_result": claim.get("verification_result"),
                    "groundedness": claim.get("groundedness"),
                    "hallucination_risk": claim.get("hallucination_risk"),
                    "citation_status": claim.get("citation_status"),
                    "evidence_ids": evidence_ids,
                    "invalid_evidence_ids": invalid_evidence_ids,
                    "explanation": claim.get("explanation"),
                    "evidence_gaps": claim.get("evidence_gaps", []),
                }
            )

        verified = sum(
            row.get("verification_result") == "verified" for row in table
        )
        review = sum(
            row.get("verification_result") == "needs_review" for row in table
        )
        rejected = sum(
            row.get("verification_result") == "rejected" for row in table
        )

        return TrustGraphOutput(
            mission_id=mission_id,
            nodes=nodes,
            edges=edges,
            claim_evidence_table=table,
            summary={
                "claim_count": len(table),
                "evidence_count": len(added_evidence),
                "invalid_evidence_references": len(added_invalid_references),
                "verified_claims": verified,
                "review_required": review,
                "rejected_claims": rejected,
            },
        )
