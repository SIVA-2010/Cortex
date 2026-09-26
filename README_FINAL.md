# CORTEX Final Completion Patch

This package completes the remaining CORTEX modules on top of the working project built through Phase 4 Step 5.

## Final capabilities

- Azure AI Mission Planner with safe template fallback
- Adaptive agent routing using capability, trust, reliability, latency and hallucination history
- End-to-end LangGraph orchestration
- Parallel knowledge retrieval and complaint intelligence
- Automatic node retries and failure recovery
- PostgreSQL-backed LangGraph checkpoints
- Human-in-the-loop pause, approve, reject and revision flow
- Workflow, agent-run, claim, evidence, governance and evaluation persistence
- TrustGraph APIs and visualization
- Mission Success Score and executive report
- Secure uploads for complaint CSV and knowledge documents
- Dashboard, audit logs and component health
- ShadowBench isolated replay and model comparison
- Premium responsive Streamlit design system
- Docker definitions, tests and PowerShell setup scripts

## Important

This is an overlay for your existing working `CORTEX/` directory. It intentionally does not include `.env`, Azure credentials, PostgreSQL passwords, the existing tested LLM/RAG services, or your generated datasets.

## Installation

1. Back up the current project.
2. Extract this ZIP directly into the existing `CORTEX/` folder.
3. Allow `backend`, `frontend`, `tests`, `docs`, `scripts`, and `.streamlit` to merge.
4. Do not create an additional nested patch folder.
5. Keep the existing `.env`; add the new fields from `.env.final.example`.

PowerShell:

```powershell
cd C:\Users\sivag\Downloads\CORTEX
.cortex\Scripts\Activate.ps1
pip install -r requirements.final.txt
python -m backend.scripts.upgrade_final_schema
python -m backend.init_db
python -m backend.scripts.setup_checkpointer
python -m compileall backend frontend
pytest
```

## Required `.env` changes

```env
WORKFLOW_CHECKPOINTER=postgres
MAX_WORKFLOW_RETRIES=2
DEFAULT_COMPLAINT_CSV=./sample_data/complaints/customer_complaints_100.csv
DEFAULT_OPERATIONS_DIR=./sample_data/operations
AZURE_OPENAI_SHADOW_DEPLOYMENT=
SHADOWBENCH_USE_MAIN_IF_UNSET=true
AZURE_OPENAI_INPUT_COST_PER_1M=0
AZURE_OPENAI_OUTPUT_COST_PER_1M=0
```

Keep your existing working Azure OpenAI, PostgreSQL, ChromaDB and JWT values. Never paste credentials into source code.

## End-to-end test

```powershell
python -m backend.scripts.test_full_workflow --auto-approve
```

Expected final output:

```text
Final status : completed
Final stage  : completed
Mission score: <0-100>
```

The workflow may first enter `awaiting_approval`; the `--auto-approve` flag exercises the resume path for local integration testing.

## Start the application

Terminal 1:

```powershell
uvicorn backend.main:app --reload
```

Terminal 2:

```powershell
streamlit run frontend/app.py
```

Open:

```text
http://localhost:8501
```

Reviewer demo account:

```text
reviewer@cortex.com
Review@123
```

## Recommended demonstration flow

1. Sign in as Reviewer.
2. Open System Status and confirm all components are ready.
3. Create a customer-complaint mission.
4. Upload the 100-row synthetic CSV or use default data.
5. Start the workflow.
6. Open Workflow Monitor and show parallel agent execution.
7. Approve the high-risk recommendation in Approval Centre.
8. Open TrustGraph and inspect claim-to-evidence lineage.
9. Open Executive Report and explain the Mission Success Score.
10. Run ShadowBench and show the isolated comparison.
11. Open Audit Logs to show traceability.

## Database additions

The upgrade script creates missing tables without dropping current data:

```text
task_dependencies
agent_runs
claims
evidence
governance_decisions
evaluation_results
workflow_approvals
uploaded_documents
shadow_runs
```

It reuses the existing `workflow_runs` and `audit_logs` tables.

## Docker (optional)

The compose file connects containers to the existing PostgreSQL installation on the Windows host.

```powershell
docker compose up --build
```

## Production deployment notes

Before a real enterprise deployment:

- Store Azure and JWT secrets in Azure Key Vault.
- Run behind HTTPS and an enterprise reverse proxy.
- Replace demo passwords and disable public demo accounts.
- Configure accurate model pricing for cost analytics.
- Use a separate shadow deployment for promotion comparisons.
- Add organization-specific data-retention and deletion policies.
- Run load, penetration, accessibility and disaster-recovery tests.
- Use managed PostgreSQL and backed-up persistent storage.
