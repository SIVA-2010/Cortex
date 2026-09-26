# Final CORTEX Architecture

```text
Streamlit Enterprise Mission Control
                 |
              FastAPI
                 |
   +-------------+--------------+
   |             |              |
PostgreSQL    ChromaDB       Azure OpenAI
   |             |              |
   +-------------+--------------+
                 |
             LangGraph
                 |
 Plan -> Route -> Retrieve + Analyse -> Root Cause
      -> Verify -> TrustGraph -> Govern -> Human Gate
      -> Evaluate -> Report -> Persist
```

## Control plane

- JWT authentication and role-based access
- Natural-language mission intake
- Agent Passport registry
- Adaptive routing
- Live workflow state and retries
- Human approval for high-risk actions
- Complete audit and decision lineage

## Data plane

- PostgreSQL: application state, workflows, agent runs, claims, approvals and audit logs
- ChromaDB: approved enterprise knowledge and semantic retrieval
- CSV/Pandas: deterministic structured analysis and PII-safe aggregation
- Azure OpenAI: reasoning, structured planning, explanation and reporting

## Trust plane

```text
Mission -> Task -> Agent Run -> Claim -> Evidence
        -> Verification -> Governance -> Approval
        -> Evaluation -> Executive Report
```

## Workflow stages

| Stage | Function |
|---|---|
| Mission Planning | Converts objective into structured tasks |
| Adaptive Routing | Selects agents using operational evidence |
| Parallel Analysis | Retrieves policies while analysing complaints |
| Root-Cause Analysis | Connects patterns to operational evidence |
| Verification | Calculates groundedness and hallucination risk |
| TrustGraph | Maps claims to evidence |
| Governance | Allows, verifies, requests approval or blocks |
| Human Approval | Pauses and resumes the workflow |
| Evaluation | Calculates the Mission Success Score |
| Reporting | Produces approved executive decision support |
