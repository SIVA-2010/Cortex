# CORTEX — Enterprise AI Mission Control

> **Autonomous multi-agent orchestration platform with adaptive routing, TrustGraph verification, and human-in-the-loop governance.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Workflows-FF6F00?style=flat)](https://github.com/langchain-ai/langgraph)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-4169E1?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange?style=flat)](https://www.trychroma.com/)

---

## Architecture Overview

```text
Streamlit Enterprise Mission Control UI
                 │
              FastAPI
                 │
   ┌─────────────┼──────────────┐
   │             │              │
PostgreSQL    ChromaDB    Azure OpenAI / LLM
   │             │              │
   └─────────────┼──────────────┘
                 │
             LangGraph
                 │
  Plan ─► Route ─► Retrieve + Analyse ─► Root Cause
       ─► Verify ─► TrustGraph ─► Govern ─► Human Gate
       ─► Evaluate ─► Report ─► Persist
```

### Core Architecture Planes

- **Control Plane:** JWT authentication, role-based access control, natural-language mission intake, agent passport registry, adaptive routing, and live workflow state monitoring.
- **Data Plane:** 
  - **PostgreSQL:** Mission records, workflows, agent runs, claims, approvals, and audit logs with state checkpointing.
  - **ChromaDB:** Approved enterprise knowledge documents and semantic vector retrieval.
  - **Pandas:** Deterministic structured analysis and PII-safe aggregation.
  - **Azure OpenAI:** Multi-agent reasoning, planning, adversarial cross-examination, and executive reporting.
- **Trust Plane:** End-to-end evidence lineage: `Mission ➔ Task ➔ Agent Run ➔ Claim ➔ Evidence ➔ Verification ➔ Governance ➔ Human Approval ➔ Quality Evaluation ➔ Executive Report`.

---

## Workflow Stages

| Stage | Name | Description |
|---|---|---|
| 1 | **Mission Planning** | Converts natural-language objectives into structured executable tasks. |
| 2 | **Adaptive Routing** | Dynamically routes tasks to specialized agents based on capability, latency, and trust history. |
| 3 | **Parallel Analysis** | Simultaneously executes policy retrieval from ChromaDB and complaint intelligence from datasets. |
| 4 | **Root-Cause Analysis** | Synthesizes operational evidence and identifies systemic operational bottlenecks. |
| 5 | **Claim Verification** | Measures groundedness, detects hallucination risk, and injects adversarial counter-claims. |
| 6 | **TrustGraph** | Generates an interactive directed graph linking claims to supporting evidence documents. |
| 7 | **Governance** | Evaluates policy rules and decides whether to allow, flag, require approval, or block. |
| 8 | **Human-in-the-Loop** | Pauses workflow for human inspection, approval, revision, or rejection before critical decisions. |
| 9 | **Quality Evaluation** | Calculates comprehensive Mission Success Score across relevance, groundedness, and reliability. |
| 10 | **Executive Report** | Generates decision-ready briefings with downloadable PDF report exports. |

---

## Quickstart

### Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Azure OpenAI or compatible LLM credentials

### 1. Environment Setup

```powershell
# Clone the repository
git clone https://github.com/SIVA-2010/Cortex.git
cd Cortex

# Create and activate virtual environment
python -m venv .cortex
.cortex\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.final.txt
```

### 2. Configure Environment Variables

Copy the example environment file and fill in your credentials:

```powershell
copy .env.example .env
```

Key variables to configure in `.env`:
- `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT`
- `JWT_SECRET_KEY`
- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`

### 3. Initialize Database & Knowledge Base

```powershell
# Upgrade schema and initialize database tables
python -m backend.scripts.upgrade_final_schema
python -m backend.init_db
python -m backend.scripts.setup_checkpointer

# Ingest sample policy documents into vector store
python -m backend.scripts.ingest_knowledge
```

### 4. Run CORTEX

**Start Backend (FastAPI):**
```powershell
.\scripts\start_backend.ps1
# Or: uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

**Start Frontend (Streamlit):**
```powershell
.\scripts\start_frontend.ps1
# Or: streamlit run frontend/app.py
```

Open your browser to `http://localhost:8501` to access Mission Control.

---

## Project Structure

```text
CORTEX/
├── backend/                  # FastAPI backend
│   ├── agents/               # Autonomous agent definitions (Planner, Router, Verification, etc.)
│   ├── graph/                # LangGraph state machine and workflow definition
│   ├── routers/              # API routes (missions, approvals, knowledge, shadowbench)
│   ├── services/             # Core business logic (LLM, vector store, TrustGraph, PII)
│   └── scripts/              # Database migration, ingestion, and test scripts
├── frontend/                 # Streamlit Mission Control UI
│   ├── app.py                # Main multi-page Streamlit application
│   └── report_pdf.py         # PDF export generator for Executive Reports
├── sample_data/              # Sample complaints, operations, and policy documents
├── docker/                   # Docker container definitions
└── docs/                     # Architecture, walkthrough, and troubleshooting guides
```

---

## License

This project is licensed under the MIT License.
