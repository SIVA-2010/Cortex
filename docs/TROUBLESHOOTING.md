# Troubleshooting

## Workflow checkpointer fails

```powershell
pip install -U langgraph langgraph-checkpoint-postgres psycopg-pool
python -m backend.scripts.setup_checkpointer
```

Confirm `WORKFLOW_CHECKPOINTER=postgres` and that the PostgreSQL credentials are the same values already used by SQLAlchemy.

## Workflow stays queued

Keep the FastAPI process running. The local prototype executes with FastAPI background tasks. Check the terminal and `GET /api/v1/workflow-runs/{run_id}`.

## Approval does not appear

The Guardian only creates approval requests for high-risk recommendations. Check the workflow status and governance artifact. Use the supplied operational evidence, which includes a high-impact rollback recommendation.

## ChromaDB is empty

```powershell
python -m backend.scripts.ingest_knowledge
```

## Root-cause stage cannot find operational evidence

```powershell
python -m backend.scripts.generate_operational_evidence
```

Confirm `DEFAULT_OPERATIONS_DIR=./sample_data/operations`.

## ShadowBench says a distinct deployment is required

Set:

```env
AZURE_OPENAI_SHADOW_DEPLOYMENT=your-shadow-deployment
```

Without a distinct deployment, CORTEX performs an isolated replay and does not recommend promotion.

## Cost remains zero

Enter the actual Azure deployment rates:

```env
AZURE_OPENAI_INPUT_COST_PER_1M=<rate>
AZURE_OPENAI_OUTPUT_COST_PER_1M=<rate>
```

## UI cannot reach FastAPI

```env
API_BASE_URL=http://127.0.0.1:8000
```

Start FastAPI before Streamlit.
