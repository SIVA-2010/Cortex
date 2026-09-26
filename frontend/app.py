from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from pydantic_settings import BaseSettings, SettingsConfigDict

PDF_REPORTS_AVAILABLE = False
PDF_REPORT_IMPORT_ERROR = ""
build_executive_report_pdf = None
build_failure_audit_pdf = None

try:
    try:
        from frontend.report_pdf import build_executive_report_pdf, build_failure_audit_pdf
    except ImportError:
        # Streamlit commonly executes frontend/app.py with frontend/ on sys.path.
        from report_pdf import build_executive_report_pdf, build_failure_audit_pdf
    PDF_REPORTS_AVAILABLE = True
except ImportError as exc:
    # PDF support is optional at application startup. This prevents the entire
    # Streamlit UI from crashing when ReportLab has not yet been installed in
    # an existing container/virtual environment. The Executive Report page
    # will show an actionable message instead.
    PDF_REPORT_IMPORT_ERROR = str(exc)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FrontendSettings(BaseSettings):
    app_name: str = "CORTEX"
    api_base_url: str = "http://127.0.0.1:8000"

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = FrontendSettings()

st.set_page_config(
    page_title="CORTEX | Enterprise AI Mission Control",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="auto",
)


NAV_ITEMS = [
    "Mission Control",
    "Create Mission",
    "Mission Plan",
    "Workflow Monitor",
    "Agent Registry",
    "Knowledge Base",
    "Approval Centre",
    "TrustGraph",
    "ShadowBench",
    "Executive Report",
    "Audit Logs",
    "System Status",
]

STAGE_LABELS = [
    ("plan_mission", "Mission Planning"),
    ("route_agents", "Adaptive Routing"),
    ("retrieve_knowledge", "Knowledge Retrieval"),
    ("analyse_complaints", "Complaint Intelligence"),
    ("analyse_root_causes", "Root-Cause Analysis"),
    ("inject_adversarial_claim", "Adversarial Agent"),
    ("verify_claims", "Claim Verification"),
    ("build_trust_graph", "TrustGraph"),
    ("apply_governance", "Governance"),
    ("request_approval", "Human Approval"),
    ("evaluate_quality", "Quality Evaluation"),
    ("generate_report", "Executive Report"),
]

LOGISTICS_STRONG_TERMS = (
    "logistics", "shipment", "shipping", "carrier", "fulfilment", "fulfillment",
    "warehouse", "dispatch", "supply chain", "freight", "last mile", "last-mile",
    "routing plan", "route optimisation", "route optimization",
)
LOGISTICS_SUPPORTING_TERMS = (
    "order", "orders", "delivery", "eta", "weight", "sku", "shipping cost",
    "promised date", "products",
)

LOGISTICS_AGENT_CODES = {
    "logistics_routing",
    "task_execution",
    "logistics_backup_execution",
    "recovery_controller",
    "recoverable_failure_agent",
    "unrecoverable_failure_agent",
}


def detect_ui_workflow_profile(mission: dict[str, Any] | None) -> str:
    """Display the workflow that belongs to the persisted Business Domain.

    Create Mission validates objective/domain compatibility before persistence, so the
    UI must not silently override the user's selected domain afterwards.
    """

    mission = mission or {}
    domain = str(mission.get("business_domain", "") or "").strip().lower()
    if "logistics" in domain:
        return "logistics"
    if "complaint" in domain:
        return "complaint"

    # Legacy-display fallback only for older records without a supported domain label.
    natural_text = " ".join(
        str(mission.get(key, "") or "")
        for key in ("title", "objective")
    ).lower()
    strong_hits = sum(term in natural_text for term in LOGISTICS_STRONG_TERMS)
    supporting_hits = sum(term in natural_text for term in LOGISTICS_SUPPORTING_TERMS)
    return "logistics" if strong_hits >= 1 or supporting_hits >= 3 else "complaint"


def _routing_agent_name(
    state: dict[str, Any],
    task_type: str,
    *,
    routing_key: str = "routing",
) -> str | None:
    for item in state.get(routing_key, []) or []:
        if isinstance(item, dict) and str(item.get("task_type")) == task_type:
            value = str(item.get("selected_agent_name", "")).strip()
            if value:
                return value
    return None


def workflow_stage_labels(
    workflow_profile: str,
    test_mode: str,
    state: dict[str, Any] | None = None,
) -> list[tuple[str, str]]:
    state = state or {}
    profile = str(workflow_profile or "complaint").lower()
    mode = str(test_mode or "normal").lower()
    if profile != "logistics":
        # Continuous adversarial assurance is part of the standard complaint path.
        # It is not activated by the wording of the business question.
        return list(STAGE_LABELS)

    primary_name = _routing_agent_name(state, "logistics_execution") or "Task Execution Agent"
    fallback_name = _routing_agent_name(
        state,
        "logistics_execution",
        routing_key="recovery_routing",
    ) or "Fallback Logistics Agent"
    stages: list[tuple[str, str]] = [
        ("plan_mission", "Mission Planner"),
        ("route_agents", "Adaptive Agent Router"),
        ("analyse_logistics", "Logistics Routing Agent"),
    ]
    if mode == "logistics_reroute_failure":
        stages.extend(
            [
                ("execute_logistics", "Recoverable Failure Agent - TEST ONLY"),
                ("recover_logistics", "Recovery Controller"),
                ("reroute_logistics", "Adaptive Router - Recovery Reroute"),
                ("execute_logistics_fallback", f"Fallback - {fallback_name}"),
            ]
        )
    elif mode == "logistics_stop_failure":
        stages.extend(
            [
                ("execute_logistics", "Unrecoverable Failure Agent - TEST ONLY"),
                ("recover_logistics", "Recovery Controller"),
                ("workflow_stopped", "Workflow Stopped"),
            ]
        )
    else:
        stages.append(("execute_logistics", primary_name))

    stages.extend(
        [
            ("verify_claims", "Verification Agent"),
            ("build_trust_graph", "TrustGraph"),
            ("apply_governance", "Guardian Governance Agent"),
            ("request_approval", "Human Approval"),
            ("evaluate_quality", "Quality Evaluator"),
            ("generate_report", "Executive Report Agent"),
        ]
    )
    return stages


def render_workflow_preview(
    workflow_profile: str,
    test_mode: str,
    state: dict[str, Any] | None = None,
) -> None:
    """Render the exact agent lane that will execute for the selected run mode."""
    profile = str(workflow_profile or "complaint").lower()
    mode = str(test_mode or "normal").lower()
    st.markdown("### Workflow for selected execution mode")
    cards = ['<div class="stage-grid">']
    for index, (stage, label) in enumerate(
        workflow_stage_labels(profile, mode, state or {}),
        start=1,
    ):
        test_only = (
            stage == "execute_logistics"
            and mode in {"logistics_reroute_failure", "logistics_stop_failure"}
        )
        recovery_stage = stage in {"recover_logistics", "reroute_logistics", "execute_logistics_fallback"}
        stop_stage = stage == "workflow_stopped"
        border = ""
        if test_only:
            border = ' style="border-color:rgba(255,127,155,.48)"'
        elif recovery_stage:
            border = ' style="border-color:rgba(255,200,107,.42)"'
        elif stop_stage:
            border = ' style="border-color:rgba(255,127,155,.58);border-style:dashed"'
        status_label = (
            "test only" if test_only
            else "recovery" if recovery_stage
            else "stop branch" if stop_stage
            else "planned"
        )
        cards.append(
            f'<div class="stage-card"{border}>'
            f'<div class="stage-index">{index:02d}</div>'
            f'<div class="stage-name">{html.escape(label)}</div>'
            f'{badge(status_label)}</div>'
        )
    cards.append('</div>')
    st.markdown("".join(cards), unsafe_allow_html=True)
    st.markdown('<div class="workflow-preview-gap" aria-hidden="true"></div>', unsafe_allow_html=True)


def apply_design_system() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {
            --bg: #050914;
            --surface: rgba(13, 22, 43, .82);
            --surface-2: rgba(18, 31, 57, .78);
            --surface-3: rgba(7, 14, 29, .92);
            --line: rgba(151, 178, 255, .14);
            --line-strong: rgba(101, 231, 255, .32);
            --text: #f7f9ff;
            --muted: #91a0bb;
            --cyan: #65e7ff;
            --blue: #7c8cff;
            --violet: #a86dff;
            --green: #73f3bd;
            --amber: #ffc86b;
            --red: #ff7f9b;
            --radius-xl: 22px;
            --radius-lg: 17px;
            --radius-md: 13px;
        }

        *, *::before, *::after { box-sizing: border-box; }
        html, body { width: 100%; max-width: 100%; min-width: 0 !important; overflow-x: hidden; }
        html, body, [class*="css"] { font-family: Inter, sans-serif; }
        #MainMenu, footer { visibility: hidden; }
        [data-testid="stToolbar"] { display: none; }
        [data-testid="stHeader"] { background: transparent; }

        .stApp {
            color: var(--text);
            overflow-x: hidden;
            background:
                radial-gradient(850px 520px at -8% -10%, rgba(70, 93, 255, .22), transparent 67%),
                radial-gradient(760px 460px at 108% 0%, rgba(0, 230, 210, .13), transparent 64%),
                radial-gradient(650px 450px at 52% 112%, rgba(155, 82, 255, .10), transparent 68%),
                var(--bg);
        }
        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            opacity: .15;
            background-image:
                linear-gradient(rgba(255,255,255,.022) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,.022) 1px, transparent 1px);
            background-size: 44px 44px;
            mask-image: linear-gradient(to bottom, black, transparent 86%);
        }

        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        main,
        .main,
        .block-container {
            min-width: 0 !important;
            max-width: 100% !important;
            overflow-x: hidden !important;
        }
        [data-testid="stMainBlockContainer"], .block-container {
            width: min(100%, 1540px) !important;
            margin-inline: auto;
            padding: 1.55rem clamp(1rem, 2.35vw, 2.35rem) 4rem !important;
        }
        [data-testid="stElementContainer"],
        [data-testid="stMarkdownContainer"],
        [data-testid="column"],
        [data-testid="stVerticalBlock"],
        [data-testid="stHorizontalBlock"] {
            min-width: 0 !important;
            max-width: 100%;
        }

        h1, h2, h3 {
            font-family: "Space Grotesk", Inter, sans-serif;
            letter-spacing: -.035em;
            overflow-wrap: normal;
            word-break: normal;
        }
        p, li, .metric-detail, .timeline-title, .timeline-meta, .finding-card,
        .component-value, .agent-description, .mission-copy {
            overflow-wrap: anywhere;
            word-break: normal;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            width: 282px !important;
            min-width: 282px !important;
            background: linear-gradient(180deg, rgba(8, 15, 31, .985), rgba(6, 11, 24, .98));
            border-right: 1px solid var(--line);
            box-shadow: 22px 0 70px rgba(0,0,0,.22);
        }
        [data-testid="stSidebarContent"] {
            padding: .9rem .8rem 1.25rem;
            overflow-x: hidden;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            min-height: 2.28rem;
            padding: .38rem .32rem !important;
            display: flex;
            align-items: center;
            line-height: 1.25;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] p {
            font-size: .91rem;
            white-space: normal;
        }

        /* Brand and headings */
        .brand-mark { display:flex; align-items:center; gap:.75rem; margin:.2rem 0 1.25rem; }
        .brand-orb {
            width:38px; height:38px; flex:0 0 38px; border-radius:13px; position:relative;
            background:linear-gradient(135deg,var(--cyan),var(--blue) 48%,var(--violet));
            box-shadow:0 0 30px rgba(101,231,255,.28), inset 0 1px 1px rgba(255,255,255,.65);
        }
        .brand-orb::after { content:""; position:absolute; inset:9px; border:2px solid rgba(5,9,20,.8); border-radius:7px; transform:rotate(45deg); }
        .brand-name { font:700 1.08rem "Space Grotesk"; letter-spacing:.20em; }
        .eyebrow { color:var(--cyan); font-size:.73rem; font-weight:800; letter-spacing:.18em; text-transform:uppercase; }
        .hero-title {
            font:700 clamp(2.4rem,4vw,4.65rem)/1 "Space Grotesk";
            letter-spacing:-.06em; margin:.72rem 0 1rem;
            background:linear-gradient(92deg,#fff 2%,#c9d7ff 48%,#7ef2df 98%);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        }
        .hero-copy { color:#a8b5ca; font-size:1.06rem; line-height:1.72; max-width:780px; }
        .page-title { font-size:clamp(2rem,3.4vw,3.45rem); line-height:1.06; margin:.42rem 0 .72rem; }
        .page-subtitle { color:var(--muted); font-size:clamp(.94rem,1.25vw,1.08rem); line-height:1.7; margin-bottom:1.55rem; max-width:880px; }

        /* Shared surfaces */
        .glass, div[data-testid="stMetric"], div[data-testid="stForm"], [data-testid="stExpander"] {
            background:linear-gradient(145deg,rgba(19,31,57,.84),rgba(9,17,34,.79));
            border:1px solid var(--line);
            border-radius:var(--radius-xl);
            box-shadow:0 20px 62px rgba(0,0,0,.22), inset 0 1px 0 rgba(255,255,255,.04);
            backdrop-filter:blur(16px);
        }
        .glass { padding:1.25rem 1.35rem; }
        .glass:hover { border-color:rgba(101,231,255,.25); transition:.2s ease; }
        div[data-testid="stForm"] { padding:1.25rem; }
        div[data-testid="stMetric"] { padding:1rem 1.05rem; min-height:116px; }
        div[data-testid="stMetric"] label { color:#8fa2bf !important; font-weight:600; }
        div[data-testid="stMetricValue"] { font-family:"Space Grotesk"; }

        .chip {
            display:inline-flex; align-items:center; gap:.42rem; max-width:100%;
            padding:.34rem .64rem; border-radius:999px; border:1px solid var(--line);
            font-size:.75rem; font-weight:700; text-transform:capitalize; white-space:nowrap;
        }
        .chip::before { content:""; width:7px; height:7px; flex:0 0 7px; border-radius:99px; background:currentColor; box-shadow:0 0 12px currentColor; }
        .chip-ready,.chip-completed,.chip-allow,.chip-approved { color:var(--green); background:rgba(115,243,189,.08); }
        .chip-running,.chip-queued { color:var(--cyan); background:rgba(101,231,255,.08); }
        .chip-awaiting_approval,.chip-pending,.chip-approval_required { color:var(--amber); background:rgba(255,200,107,.08); }
        .chip-failed,.chip-block,.chip-reject,.chip-rejected { color:var(--red); background:rgba(255,127,155,.08); }
        .chip-draft,.chip-verify,.chip-needs_review,.chip-waiting,.chip-unknown { color:#b8c6dc; background:rgba(184,198,220,.08); }
        .chip-test_only,.chip-adversarial,.chip-invalid { color:var(--amber); background:rgba(255,200,107,.09); }
        .chip-skipped { color:#708099; background:rgba(112,128,153,.08); }
        .chip-missing,.chip-stop { color:var(--red); background:rgba(255,127,155,.08); }
        .chip-reroute,.chip-rerouted { color:var(--cyan); background:rgba(101,231,255,.08); }

        .metric-label { color:var(--muted); font-size:.73rem; letter-spacing:.09em; line-height:1.35; text-transform:uppercase; }
        .metric-value {
            font:700 clamp(1.45rem,2.05vw,2.05rem)/1.08 "Space Grotesk";
            margin:.48rem 0 .34rem;
            overflow-wrap:normal !important;
            word-break:keep-all !important;
            hyphens:none;
            white-space:normal;
        }
        .metric-detail { color:#8796af; font-size:.82rem; line-height:1.55; }
        .divider { height:1px; background:linear-gradient(90deg,transparent,var(--line),transparent); margin:1.15rem 0; }

        /* True responsive HTML grids. These replace fragile six-column Streamlit rows. */
        .metric-grid {
            display:grid;
            grid-template-columns:repeat(auto-fit,minmax(min(100%,var(--metric-min,210px)),1fr));
            gap:.85rem;
            width:100%;
            margin:.2rem 0 1.35rem;
        }
        .metric-card {
            min-width:0;
            min-height:126px;
            padding:1.12rem 1.12rem 1rem;
            background:linear-gradient(145deg,rgba(19,31,57,.84),rgba(9,17,34,.80));
            border:1px solid var(--line);
            border-radius:18px;
            box-shadow:0 16px 44px rgba(0,0,0,.18),inset 0 1px 0 rgba(255,255,255,.035);
        }
        .metric-card:hover { border-color:var(--line-strong); }

        /* Targeted responsive layouts */
        .login-page-marker { display:none !important; }
        [data-testid="stAppViewContainer"]:has(.login-page-marker) [data-testid="stHeader"],
        [data-testid="stAppViewContainer"]:has(.login-page-marker) [data-testid="stSidebarCollapsedControl"],
        [data-testid="stAppViewContainer"]:has(.login-page-marker) [data-testid="collapsedControl"] {
            display:none !important;
            width:0 !important;
            min-width:0 !important;
            height:0 !important;
            min-height:0 !important;
            padding:0 !important;
            margin:0 !important;
            border:0 !important;
            box-shadow:none !important;
        }
        .login-form-shell { max-width:430px; margin:0; padding-top:1.15rem; }
        .login-promo-shell { padding-top:1.15rem; }

        .blueprint-grid {
            display:grid;
            grid-template-columns:repeat(4,minmax(0,1fr));
            gap:.82rem;
            width:100%;
            margin:1rem 0 1.35rem;
        }
        .blueprint-card {
            min-width:0;
            padding:1rem;
            border:1px solid var(--line);
            border-radius:16px;
            background:linear-gradient(145deg,rgba(18,31,57,.82),rgba(7,14,29,.86));
        }
        .blueprint-card .stage-name { margin:.35rem 0 .45rem; }

        .activity-grid {
            display:grid;
            grid-template-columns:repeat(auto-fit,minmax(min(100%,230px),1fr));
            gap:.8rem;
            width:100%;
            margin:.35rem 0 1rem;
        }
        .activity-card {
            min-width:0;
            padding:1rem 1.05rem;
            border:1px solid var(--line);
            border-radius:16px;
            background:rgba(9,17,34,.68);
        }
        .mission-action-block { display:grid; gap:.55rem; align-content:center; justify-items:stretch; }
        .mission-action-block .chip { justify-self:start; }
        .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
            white-space:nowrap !important;
            word-break:keep-all !important;
            overflow-wrap:normal !important;
        }

        .component-grid {
            display:grid;
            grid-template-columns:repeat(auto-fit,minmax(min(100%,245px),1fr));
            gap:.9rem;
            width:100%;
            margin-top:1.1rem;
        }
        .component-card {
            min-width:0;
            padding:1.15rem;
            border:1px solid var(--line);
            border-radius:18px;
            background:linear-gradient(145deg,rgba(18,31,57,.82),rgba(7,14,29,.88));
        }
        .component-title { font:700 1.25rem "Space Grotesk"; margin:.45rem 0 .7rem; }
        .component-details { display:grid; gap:.45rem; margin-top:.8rem; }
        .component-row { display:grid; grid-template-columns:minmax(84px,.72fr) minmax(0,1.28fr); gap:.65rem; align-items:start; font-size:.79rem; }
        .component-key { color:#71839f; text-transform:capitalize; }
        .component-value { color:#a9b7cd; text-align:right; }

        .agent-grid {
            display:grid;
            grid-template-columns:repeat(3,minmax(0,1fr));
            gap:.9rem;
            width:100%;
            margin-top:1rem;
        }
        .agent-card {
            min-width:0;
            min-height:290px;
            display:flex;
            flex-direction:column;
            padding:1.25rem;
            border:1px solid var(--line);
            border-radius:20px;
            background:linear-gradient(145deg,rgba(18,31,57,.84),rgba(8,16,32,.82));
        }
        .agent-card:hover { border-color:rgba(101,231,255,.30); }
        .agent-card h3 { margin:.65rem 0 .75rem; font-size:1.45rem; }
        .agent-description { color:#8fa0ba; line-height:1.65; min-height:3.35rem; }
        .agent-capabilities { color:#dce6f7; line-height:1.65; margin:.35rem 0 1rem; }
        .agent-badges { margin-top:auto; display:flex; gap:.45rem; flex-wrap:wrap; }

        /* Timeline, stages, reports */
        .timeline { position:relative; padding-left:1.18rem; }
        .timeline::before { content:""; position:absolute; left:5px; top:8px; bottom:8px; width:1px; background:linear-gradient(var(--cyan),rgba(124,140,255,.18)); }
        .timeline-item { position:relative; margin:0 0 .85rem; padding:.9rem 1rem; border:1px solid var(--line); border-radius:14px; background:rgba(9,17,34,.62); }
        .timeline-item::before { content:""; position:absolute; left:-1.26rem; top:1.2rem; width:9px; height:9px; border-radius:50%; background:var(--cyan); box-shadow:0 0 16px var(--cyan); }
        .timeline-meta { display:flex; justify-content:space-between; gap:.85rem; color:#8292ac; font-size:.75rem; }
        .timeline-title { font-weight:700; margin:.2rem 0; }

        .stage-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,185px),1fr)); gap:.75rem; width:100%; }
        .workflow-preview-gap { height:clamp(1rem,1.7vw,1.55rem); width:100%; }
        .stage-card { min-width:0; border:1px solid var(--line); border-radius:16px; padding:.92rem; background:rgba(10,18,36,.70); min-height:110px; }
        .stage-card.active { border-color:rgba(101,231,255,.55); box-shadow:0 0 24px rgba(101,231,255,.11); }
        .stage-card.done { border-color:rgba(115,243,189,.32); }
        .stage-card.failed { border-color:rgba(255,127,155,.58); box-shadow:0 0 24px rgba(255,127,155,.08); }
        .stage-card.skipped { opacity:.62; border-style:dashed; }
        .stage-index { color:var(--muted); font-size:.7rem; }
        .stage-name { font-weight:700; margin:.42rem 0; line-height:1.35; display:flex; align-items:center; gap:.45rem; }
        .stage-status-icon { width:9px; height:9px; flex:0 0 9px; border-radius:50%; background:#708099; }
        .stage-status-icon.completed { background:var(--green); box-shadow:0 0 12px rgba(115,243,189,.72); }
        .stage-status-icon.running, .stage-status-icon.active { background:var(--cyan); box-shadow:0 0 12px rgba(101,231,255,.72); }
        .stage-status-icon.failed { background:var(--red); box-shadow:0 0 14px rgba(255,127,155,.95); }
        .stage-status-icon.skipped, .stage-status-icon.waiting { background:#708099; box-shadow:none; }
        .timeline-item.timeline-failed::before { background:var(--red); box-shadow:0 0 16px rgba(255,127,155,.95); }
        .timeline-item.timeline-completed::before { background:var(--green); box-shadow:0 0 16px rgba(115,243,189,.72); }

        .report-section { border-left:2px solid var(--cyan); padding-left:1rem; margin:1.25rem 0; }
        .finding-card { padding:1rem; border:1px solid var(--line); border-radius:15px; background:rgba(10,18,36,.68); margin:.65rem 0; }
        .score-ring { width:130px; height:130px; border-radius:50%; display:grid; place-items:center; margin:auto; background:conic-gradient(var(--green) var(--score),rgba(255,255,255,.08) 0); position:relative; }
        .score-ring::after { content:""; position:absolute; inset:10px; border-radius:50%; background:#0a1224; }
        .score-text { z-index:1; font:700 1.75rem "Space Grotesk"; }

        /* Inputs and buttons */
        .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
            min-height:2.85rem; max-width:100%; border-radius:12px; font-weight:750;
            border:1px solid rgba(124,140,255,.34);
            background:linear-gradient(135deg,rgba(83,106,255,.96),rgba(125,83,255,.96));
            color:white; box-shadow:0 10px 26px rgba(72,76,220,.22); transition:.2s ease;
        }
        .stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {
            transform:translateY(-1px); border-color:rgba(101,231,255,.65); box-shadow:0 14px 34px rgba(72,76,220,.31);
        }
        .stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] > div, .stNumberInput input {
            background:rgba(5,11,24,.78) !important;
            border-color:var(--line) !important;
            border-radius:11px !important;
        }
        .stProgress > div > div > div { background:linear-gradient(90deg,var(--blue),var(--cyan),var(--green)); }

        /* Dataframes and charts */
        [data-testid="stDataFrame"], [data-testid="stDataFrameResizable"] {
            width:100% !important; max-width:100% !important; min-width:0 !important;
            border:1px solid var(--line); border-radius:15px; overflow:auto !important;
            -webkit-overflow-scrolling:touch;
        }
        [data-testid="stPlotlyChart"], [data-testid="stPlotlyChart"] > div,
        .js-plotly-plot, .plot-container, .svg-container {
            width:100% !important; max-width:100% !important; min-width:0 !important;
        }
        [data-testid="stImage"] img, [data-testid="stImage"] svg,
        [data-testid="stVideo"] video, iframe { max-width:100% !important; height:auto; }
        [data-testid="stFileUploader"], [data-testid="stFileUploaderDropzone"],
        [data-testid="stForm"], [data-testid="stExpander"] { width:100%; max-width:100%; min-width:0; }
        pre, code { white-space:pre-wrap; overflow-wrap:anywhere; }

        .login-shell { padding-top:7vh; }
        .ambient-orb { position:fixed; width:330px; height:330px; border-radius:50%; filter:blur(90px); opacity:.11; pointer-events:none; }
        .orb-a { left:20%; top:14%; background:var(--blue); }
        .orb-b { right:8%; bottom:5%; background:var(--cyan); }

        .layout-marker { display:none !important; width:0 !important; height:0 !important; margin:0 !important; padding:0 !important; }

        /* Small desktop / tablet landscape: stack only the big split layouts.
           Do not globally wrap every Streamlit column. */
        @media (max-width: 1180px) {
            [data-testid="stMainBlockContainer"], .block-container { padding:1.25rem 1.25rem 3rem !important; }
            .agent-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
            [data-testid="stHorizontalBlock"]:has(.layout-login-split),
            [data-testid="stHorizontalBlock"]:has(.layout-dashboard-split),
            [data-testid="stHorizontalBlock"]:has(.layout-create-split),
            [data-testid="stHorizontalBlock"]:has(.layout-workflow-split),
            [data-testid="stHorizontalBlock"]:has(.layout-knowledge-split),
            [data-testid="stHorizontalBlock"]:has(.layout-shadow-split),
            [data-testid="stHorizontalBlock"]:has(.layout-report-split) {
                display:flex !important;
                flex-direction:column !important;
                gap:1rem !important;
            }
            [data-testid="stHorizontalBlock"]:has(.layout-login-split) > [data-testid="column"],
            [data-testid="stHorizontalBlock"]:has(.layout-dashboard-split) > [data-testid="column"],
            [data-testid="stHorizontalBlock"]:has(.layout-create-split) > [data-testid="column"],
            [data-testid="stHorizontalBlock"]:has(.layout-workflow-split) > [data-testid="column"],
            [data-testid="stHorizontalBlock"]:has(.layout-knowledge-split) > [data-testid="column"],
            [data-testid="stHorizontalBlock"]:has(.layout-shadow-split) > [data-testid="column"],
            [data-testid="stHorizontalBlock"]:has(.layout-report-split) > [data-testid="column"] {
                flex:1 1 100% !important; width:100% !important; min-width:0 !important; max-width:100% !important;
            }
            .login-shell { padding-top:1rem; }
        }

        @media (max-width: 1180px) {
            .blueprint-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
        }

        @media (max-width: 980px) {
            .metric-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
            .metric-value { font-size:clamp(1.45rem,3.6vw,1.9rem); }
        }

        @media (max-width: 860px) {
            .login-form-shell, .login-promo-shell { max-width:none; padding-top:.35rem; }
        }

        /* Tablet / phone */
        @media (max-width: 760px) {
            section[data-testid="stSidebar"][aria-expanded="true"] {
                width:min(86vw,300px) !important;
                min-width:min(86vw,300px) !important;
            }
            [data-testid="stMainBlockContainer"], .block-container {
                width:100% !important;
                padding:.9rem .78rem 2.5rem !important;
            }
            [data-testid="stHorizontalBlock"] {
                display:flex !important;
                flex-direction:column !important;
                flex-wrap:nowrap !important;
                gap:.72rem !important;
                width:100% !important;
            }
            [data-testid="stHorizontalBlock"] > [data-testid="column"] {
                flex:1 1 100% !important;
                width:100% !important;
                min-width:0 !important;
                max-width:100% !important;
            }
            .hero-title { font-size:clamp(2rem,10vw,2.75rem); line-height:1.05; letter-spacing:-.05em; }
            .hero-title br { display:none; }
            .hero-copy { font-size:.94rem; line-height:1.58; }
            .page-title { font-size:clamp(1.8rem,8.4vw,2.25rem); line-height:1.08; }
            .page-subtitle { font-size:.92rem; line-height:1.56; margin-bottom:1.15rem; }
            h1 { font-size:clamp(1.8rem,8vw,2.25rem) !important; }
            h2 { font-size:clamp(1.45rem,6.5vw,1.85rem) !important; }
            h3 { font-size:clamp(1.15rem,5.2vw,1.45rem) !important; }
            .glass, div[data-testid="stMetric"], div[data-testid="stForm"], [data-testid="stExpander"] { border-radius:15px; }
            .glass, div[data-testid="stForm"] { padding:.95rem; }
            .metric-grid, .component-grid, .agent-grid, .blueprint-grid, .activity-grid { grid-template-columns:1fr; gap:.72rem; }
            .metric-card { min-height:auto; padding:.95rem; }
            .agent-card { min-height:auto; }
            .stage-grid { grid-template-columns:repeat(auto-fit,minmax(min(100%,145px),1fr)); gap:.65rem; }
            .timeline-meta { flex-direction:column; align-items:flex-start; gap:.2rem; }
            .score-ring { width:106px; height:106px; }
            .score-ring::after { inset:8px; }
            .score-text { font-size:1.4rem; }
            .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button { width:100%; min-height:3rem; }
            [data-testid="stFileUploaderDropzone"] { min-height:92px !important; padding:.75rem !important; }
            .js-plotly-plot .plotly .modebar { display:none !important; }
            .component-row { grid-template-columns:1fr; gap:.12rem; }
            .component-value { text-align:left; }
        }

        @media (max-width: 480px) {
            [data-testid="stMainBlockContainer"], .block-container { padding:.72rem .58rem 2rem !important; }
            .eyebrow { font-size:.66rem; letter-spacing:.14em; }
            .page-title { font-size:1.75rem; }
            .metric-label { font-size:.67rem; }
            .metric-value { font-size:1.55rem; }
            .metric-detail { font-size:.77rem; }
            .glass, .metric-card, .component-card, .agent-card { border-radius:13px; }
            .stage-grid { grid-template-columns:1fr; }
            .workflow-preview-gap { height:.9rem; }
            .chip { font-size:.69rem; padding:.29rem .52rem; }
            [data-testid="stDataFrame"], [data-testid="stDataFrameResizable"] { border-radius:11px; }
        }

        @media (orientation: landscape) and (max-height:560px) {
            .login-shell { padding-top:0; }
            .ambient-orb { display:none; }
            [data-testid="stMainBlockContainer"], .block-container { padding-top:.65rem !important; }
        }

        @media (prefers-reduced-motion:reduce) {
            *,*::before,*::after { animation-duration:.01ms!important; animation-iteration-count:1!important; transition-duration:.01ms!important; scroll-behavior:auto!important; }
        }
        </style>
        <div class="ambient-orb orb-a"></div><div class="ambient-orb orb-b"></div>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    defaults = {
        "access_token": None,
        "user": None,
        "navigation": "Mission Control",
        "selected_mission_id": None,
        "selected_run_id": None,
        "complaint_document_id": None,
        "complaint_document_mission_id": None,
        "workflow_execution_mode": "Continuous Adversarial Assurance - standard complaint workflow",
        "create_mission_assistant_response": None,
        "create_mission_assistant_intent": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session() -> None:
    for key in list(st.session_state):
        del st.session_state[key]


def api_request(
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout: int = 45,
) -> Any:
    headers = {"Accept": "application/json"}
    if st.session_state.access_token:
        headers["Authorization"] = f"Bearer {st.session_state.access_token}"
    try:
        response = requests.request(
            method,
            f"{settings.api_base_url.rstrip('/')}{path}",
            json=payload,
            params=params,
            headers=headers,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError("CORTEX backend is unavailable.") from exc
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", "Request failed.")
        except ValueError:
            detail = response.text or "Request failed."
        if isinstance(detail, list):
            detail = "; ".join(str(item.get("msg", item)) for item in detail)
        raise RuntimeError(str(detail))
    if not response.content:
        return {}
    return response.json()


def api_upload(path: str, uploaded_file: Any, *, mission_id: str | None = None) -> Any:
    headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
    data = {"mission_id": mission_id} if mission_id else {}
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    try:
        response = requests.post(
            f"{settings.api_base_url.rstrip('/')}{path}",
            headers=headers,
            data=data,
            files=files,
            timeout=90,
        )
    except requests.RequestException as exc:
        raise RuntimeError("Upload failed because the backend is unavailable.") from exc
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", "Upload failed.")
        except ValueError:
            detail = response.text or "Upload failed."
        raise RuntimeError(str(detail))
    return response.json()


def badge(status: str) -> str:
    value = str(status or "unknown").lower().replace(" ", "_")
    return f'<span class="chip chip-{html.escape(value)}">{html.escape(str(status).replace("_", " "))}</span>'


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="eyebrow">{html.escape(eyebrow)}</div>'
        f'<h1 class="page-title">{html.escape(title)}</h1>'
        f'<div class="page-subtitle">{html.escape(subtitle)}</div>',
        unsafe_allow_html=True,
    )


def layout_marker(name: str) -> None:
    """Place an invisible marker used by targeted responsive CSS."""
    st.markdown(f'<span class="layout-marker layout-{html.escape(name)}"></span>', unsafe_allow_html=True)


def render_metric_grid(
    items: list[tuple[str, Any, str]],
    *,
    min_width: int = 205,
) -> None:
    cards: list[str] = []
    for label, value, detail in items:
        cards.append(
            '<div class="metric-card">'
            f'<div class="metric-label">{html.escape(str(label))}</div>'
            f'<div class="metric-value">{html.escape(str(value))}</div>'
            f'<div class="metric-detail">{html.escape(str(detail))}</div>'
            '</div>'
        )
    st.markdown(
        f'<div class="metric-grid" style="--metric-min:{int(min_width)}px">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def render_login() -> None:
    # This marker removes Streamlit's empty collapsed-sidebar/header surface on the login page only.
    st.markdown('<span class="login-page-marker" aria-hidden="true"></span>', unsafe_allow_html=True)

    form_col, promo_col = st.columns([0.82, 1.18], gap="large", vertical_alignment="top")
    with form_col:
        st.markdown('<div class="login-shell login-form-shell">', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">Secure Workspace</div><h2>Welcome to CORTEX</h2>', unsafe_allow_html=True)
        st.caption("Authenticate to enter Enterprise AI Mission Control.")
        with st.form("login", clear_on_submit=False):
            email = st.text_input("Business email", value="reviewer@cortex.com")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Enter Mission Control", use_container_width=True)
        if submit:
            try:
                with st.spinner("Establishing secure session..."):
                    result = api_request(
                        "POST",
                        "/api/v1/auth/login",
                        payload={"email": email.strip(), "password": password},
                    )
                st.session_state.access_token = result["access_token"]
                st.session_state.user = result["user"]
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))
        st.caption("Demo reviewer · reviewer@cortex.com / Review@123")
        st.markdown('</div>', unsafe_allow_html=True)

    with promo_col:
        st.markdown(
            """
            <div class="login-shell login-promo-shell">
              <div class="eyebrow">Adaptive AgentOps Platform</div>
              <div class="hero-title">The control tower<br>for enterprise AI.</div>
              <div class="hero-copy">Plan, route, execute, verify and govern a digital workforce of specialist AI agents—while every claim remains traceable to evidence.</div>
              <div style="height:1.5rem"></div>
              <div class="stage-grid">
                <div class="stage-card"><div class="stage-index">01</div><div class="stage-name">Multi-Agent Orchestration</div><div class="metric-detail">LangGraph state, routing and recovery.</div></div>
                <div class="stage-card"><div class="stage-index">02</div><div class="stage-name">Decision Assurance</div><div class="metric-detail">Verification, TrustGraph and governance.</div></div>
                <div class="stage-card"><div class="stage-index">03</div><div class="stage-name">Human Oversight</div><div class="metric-detail">Approval gates for high-impact actions.</div></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sidebar() -> str:
    user = st.session_state.user
    with st.sidebar:
        st.markdown(
            '<div class="brand-mark"><div class="brand-orb"></div><div><div class="brand-name">CORTEX</div><div class="metric-detail">Enterprise AgentOps</div></div></div>',
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Workspace",
            NAV_ITEMS,
            label_visibility="collapsed",
            key="navigation",
        )
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(f"**{html.escape(user['full_name'])}**")
        st.caption(user["email"])
        st.markdown(badge(user["role"]), unsafe_allow_html=True)
        st.write("")
        if st.button("Sign out", use_container_width=True):
            reset_session()
            st.rerun()
    return page


def load_missions() -> list[dict[str, Any]]:
    try:
        return api_request("GET", "/api/v1/missions")
    except RuntimeError:
        return []


def mission_selector(missions: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    if not missions:
        st.info("Create a mission first.")
        return None
    labels = {
        f"{item['title']} · {str(item['status']).replace('_',' ').title()} · {str(item['id'])[:8]}": item
        for item in missions
    }
    options = list(labels)
    default_index = 0
    selected_id = st.session_state.get("selected_mission_id")
    if selected_id:
        for index, label in enumerate(options):
            if str(labels[label]["id"]) == str(selected_id):
                default_index = index
                break
    selected = st.selectbox("Mission", options, index=default_index, key=key)
    mission = labels[selected]
    st.session_state.selected_mission_id = mission["id"]
    return mission


def render_dashboard() -> None:
    page_header(
        "Mission Control",
        "Enterprise AI at a glance",
        "A unified view of mission performance, trust, governance and agent operations.",
    )
    try:
        summary = api_request("GET", "/api/v1/dashboard/summary")
    except RuntimeError as exc:
        st.error(str(exc))
        return

    metrics = [
        ("Active Missions", summary["active_missions"], "Live orchestration"),
        ("Completed", summary["completed_missions"], "Verified outputs"),
        ("AI Agents", summary["available_agents"], "Available workforce"),
        ("Trust Score", f"{summary['average_trust_score']:.1f}", "Registry average"),
        ("Pending Approval", summary["pending_approvals"], "Human oversight"),
        ("Mission Score", f"{summary['average_mission_score']:.1f}", "Quality average"),
    ]
    render_metric_grid(metrics, min_width=190)

    st.subheader("Recent missions")
    missions = load_missions()[:8]
    if not missions:
        st.info("No missions yet. Create the first enterprise objective.")
    else:
        for mission in missions:
            with st.container(border=True):
                info_col, action_col = st.columns([4.6, 1.25], gap="medium", vertical_alignment="center")
                with info_col:
                    st.markdown(f"**{html.escape(str(mission['title']))}**")
                    st.caption(
                        f"{mission['business_domain']} · {mission['task_count']} tasks"
                    )
                with action_col:
                    st.markdown('<div class="mission-action-block">', unsafe_allow_html=True)
                    st.markdown(badge(mission["status"]), unsafe_allow_html=True)
                    if st.button("Open", key=f"open_{mission['id']}", use_container_width=True):
                        st.session_state.selected_mission_id = mission["id"]
                        st.session_state.navigation = "Workflow Monitor"
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.subheader("Live activity")
    activity = summary.get("recent_activity", [])
    if not activity:
        st.caption("Activity will appear as workflows execute.")
    else:
        activity_cards: list[str] = []
        for item in activity[:8]:
            title = html.escape(str(item.get("action", "activity")).replace(".", " ").title())
            entity = html.escape(str(item.get("entity_type") or "CORTEX"))
            activity_cards.append(
                '<article class="activity-card">'
                f'<div class="timeline-title">{title}</div>'
                f'<div class="metric-detail">{entity}</div>'
                '</article>'
            )
        st.markdown(f'<div class="activity-grid">{"".join(activity_cards)}</div>', unsafe_allow_html=True)

    render_metric_grid(
        [
            (
                "AI Consumption",
                f"{summary['total_tokens']:,}",
                f"Tokens across {summary['total_workflow_runs']} workflow run(s) · "
                f"Estimated cost {summary.get('total_estimated_cost', 0):.4f}",
            )
        ],
        min_width=280,
    )


def render_create_mission() -> None:
    page_header(
        "New Mission",
        "Turn an objective into an AI operation",
        "Describe the outcome. CORTEX first understands the request, then creates a workflow only for supported enterprise missions.",
    )

    with st.form("create_mission"):
        title = st.text_input("Mission title", value="July Customer Experience Intelligence")
        objective = st.text_area(
            "Business objective",
            value=(
                "Analyse the synthetic customer complaints, identify recurring issues and "
                "evidence-backed root causes, verify every material claim, apply privacy and "
                "governance controls, and prepare prioritised executive recommendations."
            ),
            height=170,
            help=(
                "You can also greet CORTEX or ask about its capabilities here. "
                "For a business mission, the objective must match the selected Business Domain."
            ),
        )
        c1, c2 = st.columns(2)
        domain = c1.selectbox(
            "Business domain",
            [
                "Customer Complaint Intelligence",
                "Logistics Intelligence",
                "IT Incident Analysis",
                "Compliance Reporting",
                "Contract Review",
            ],
        )
        priority = c2.selectbox("Priority", ["high", "critical", "medium", "low"])
        c3, c4 = st.columns(2)
        risk = c3.selectbox("Risk tolerance", ["balanced", "conservative", "experimental"])
        budget = c4.number_input("Maximum AI budget", min_value=0.0, value=25.0, step=5.0)
        c5, c6 = st.columns(2)
        output_format = c5.selectbox("Output", ["executive_report", "detailed_report", "json"])
        human_approval = c6.toggle("Human approval for high risk", value=True)
        auto_plan = st.toggle("Generate initial workflow plan", value=True)
        submitted = st.form_submit_button("Create Mission", use_container_width=True)

    if submitted:
        st.session_state.create_mission_assistant_response = None
        st.session_state.create_mission_assistant_intent = None
        try:
            with st.spinner("CORTEX is understanding your request..."):
                result = api_request(
                    "POST",
                    "/api/v1/missions/submit",
                    payload={
                        "title": title,
                        "objective": objective,
                        "business_domain": domain,
                        "priority": priority,
                        "risk_tolerance": risk,
                        "max_budget": budget,
                        "output_format": output_format,
                        "human_approval_preference": human_approval,
                        "auto_plan": auto_plan,
                    },
                )

            if not bool(result.get("mission_created")):
                st.session_state.selected_mission_id = None
                st.session_state.complaint_document_id = None
                st.session_state.complaint_document_mission_id = None
                st.session_state.create_mission_assistant_response = str(
                    result.get("message") or "CORTEX could not classify the request."
                )
                st.session_state.create_mission_assistant_intent = str(
                    result.get("intent") or "unsupported"
                )
            else:
                mission = result.get("mission") or {}
                st.session_state.create_mission_assistant_response = None
                st.session_state.create_mission_assistant_intent = None
                st.session_state.selected_mission_id = mission["id"]
                st.session_state.complaint_document_id = None
                st.session_state.complaint_document_mission_id = None
                detected = str(result.get("workflow_profile") or detect_ui_workflow_profile(mission))
                st.session_state.workflow_execution_mode = (
                    "Normal Logistics"
                    if detected == "logistics"
                    else "Continuous Adversarial Assurance - standard complaint workflow"
                )
                st.success(str(result.get("message") or "Mission created successfully."))
        except RuntimeError as exc:
            st.error(str(exc))

    assistant_message = st.session_state.get("create_mission_assistant_response")
    assistant_intent = str(st.session_state.get("create_mission_assistant_intent") or "")
    if assistant_message:
        st.markdown("### CORTEX")
        if assistant_intent in {"unsupported", "domain_mismatch"}:
            st.warning(str(assistant_message))
        else:
            st.info(str(assistant_message))
        st.caption("No mission was created and the multi-agent workflow was not started.")
        return

    st.markdown("### Execution blueprint")
    st.markdown(
        """
        <div class="blueprint-grid">
          <article class="blueprint-card"><div class="stage-index">01</div><div class="stage-name">Plan</div><div class="metric-detail">CORTEX resolves the mission domain and executable task graph.</div></article>
          <article class="blueprint-card"><div class="stage-index">02</div><div class="stage-name">Route & Execute</div><div class="metric-detail">Specialists are selected by capability, trust and reliability.</div></article>
          <article class="blueprint-card"><div class="stage-index">03</div><div class="stage-name">Verify & Recover</div><div class="metric-detail">Claims are verified and controlled failures enter the recovery lane.</div></article>
          <article class="blueprint-card"><div class="stage-index">04</div><div class="stage-name">Govern & Decide</div><div class="metric-detail">Only safe, approved findings reach the executive output.</div></article>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.selected_mission_id:
        return

    try:
        selected_mission = api_request(
            "GET", f"/api/v1/missions/{st.session_state.selected_mission_id}"
        )
        readiness = api_request(
            "GET",
            f"/api/v1/missions/{st.session_state.selected_mission_id}/data-readiness",
        )
    except RuntimeError as exc:
        st.error(str(exc))
        return

    profile = str(readiness.get("workflow_profile") or detect_ui_workflow_profile(selected_mission))
    st.markdown(
        f'<div class="glass"><div class="metric-label">Detected workflow profile</div>'
        f'<div class="metric-value" style="font-size:1.15rem">'
        f'{"Logistics Intelligence" if profile == "logistics" else "Customer Complaint Intelligence"}</div>'
        f'<div class="metric-detail">The objective was validated against this selected Business Domain before the mission was created.</div></div>',
        unsafe_allow_html=True,
    )

    if profile == "complaint":
        st.markdown("### Required source dataset")
        st.caption(
            "Upload the complaint CSV containing the authoritative source records. "
            "Prompt-only complaint analysis remains disabled."
        )
        if readiness.get("ready") and readiness.get("document_id"):
            st.session_state.complaint_document_id = readiness["document_id"]
            st.session_state.complaint_document_mission_id = str(
                st.session_state.selected_mission_id
            )
        uploaded = st.file_uploader(
            "Upload source CSV",
            type=["csv"],
            key="complaint_upload",
            help="A mission-linked complaint CSV is mandatory before complaint analysis can execute.",
        )
        if uploaded and st.button("Securely upload CSV", use_container_width=True):
            try:
                with st.spinner("Validating and storing the dataset..."):
                    result = api_upload(
                        "/api/v1/uploads/complaints",
                        uploaded,
                        mission_id=str(st.session_state.selected_mission_id),
                    )
                st.session_state.complaint_document_id = result["document_id"]
                st.session_state.complaint_document_mission_id = str(
                    st.session_state.selected_mission_id
                )
                readiness["ready"] = True
                readiness["file_name"] = result["file_name"]
                st.success(f"{result['file_name']} is validated and ready for execution.")
            except RuntimeError as exc:
                st.error(str(exc))

        dataset_ready = bool(readiness.get("ready")) and bool(
            st.session_state.complaint_document_id
        )
        validation_error = str(readiness.get("validation_error") or "").strip()
        if not dataset_ready:
            if validation_error:
                st.error(validation_error)
                st.caption(
                    "Execution is blocked before Mission Planning, so no workflow agents or Azure model calls run on incompatible data."
                )
            else:
                st.warning(
                    "Dataset Required — This mission requires source data before agent "
                    "execution can begin. Upload the required dataset to continue."
                )
        else:
            validation_summary = readiness.get("validation_summary") or {}
            valid_rows = validation_summary.get("usable_row_count")
            row_detail = (
                f" · {int(valid_rows)} usable rows"
                if isinstance(valid_rows, (int, float))
                else ""
            )
            st.markdown(
                '<div class="glass"><div class="metric-label">Dataset validation gate</div>'
                '<div class="metric-value" style="font-size:1.15rem">Complaint dataset validated</div>'
                f'<div class="metric-detail">{html.escape(str(readiness.get("file_name") or "Mission-linked CSV"))} passed schema and semantic checks{html.escape(row_detail)}. Execution is enabled.</div></div>',
                unsafe_allow_html=True,
            )

        # No complaint-mode selector is shown: continuous adversarial assurance is
        # part of the standard Customer Complaint Intelligence architecture.
        test_mode = "adversarial"
        render_workflow_preview("complaint", test_mode)
    else:
        st.markdown("### Required logistics data package")
        st.caption(
            "CORTEX uses the three-file synthetic logistics package below. All three files "
            "must exist before this workflow can start."
        )
        files_ready = set(readiness.get("present_files", []))
        cards = []
        for file_name in ["orders.csv", "products.csv", "carriers.json"]:
            status_value = "ready" if file_name in files_ready else "missing"
            cards.append(
                '<div class="metric-card">'
                f'<div class="metric-label">Required file</div><div class="timeline-title">{html.escape(file_name)}</div>'
                f'{badge(status_value)}'
                f'<div class="metric-detail">sample_data/logistics/{html.escape(file_name)}</div></div>'
            )
        st.markdown(
            '<div class="metric-grid" style="--metric-min:210px">'
            + "".join(cards)
            + "</div>",
            unsafe_allow_html=True,
        )
        dataset_ready = bool(readiness.get("ready"))
        validation_error = str(readiness.get("validation_error") or "").strip()
        if dataset_ready:
            st.success("Logistics data package passed schema and data-quality validation and is ready.")
            st.caption(
                "CORTEX confirmed compatible orders, products and carrier coverage before enabling execution."
            )
        elif validation_error:
            st.error(validation_error)
            st.caption(
                "Execution is blocked before Mission Planning, so no logistics agents or Azure model calls run on incompatible data."
            )
        else:
            missing = ", ".join(readiness.get("missing_files", [])) or "required files"
            st.error(
                "Logistics Data Required — add orders.csv, products.csv and carriers.json "
                f"to sample_data/logistics. Missing: {missing}."
            )

        mode_options = [
            "Normal Logistics",
            "Recoverable Failure — Reroute",
            "Unrecoverable Failure — Stop",
        ]
        if st.session_state.get("workflow_execution_mode") not in mode_options:
            st.session_state.workflow_execution_mode = mode_options[0]

        st.markdown("### Execution mode")
        execution_mode_label = st.selectbox(
            "Choose how this workflow run should execute",
            mode_options,
            key="workflow_execution_mode",
            help="Logistics failure tests are deterministic and apply only to this workflow run.",
        )
        if execution_mode_label.startswith("Recoverable Failure"):
            test_mode = "logistics_reroute_failure"
        elif execution_mode_label.startswith("Unrecoverable Failure"):
            test_mode = "logistics_stop_failure"
        else:
            test_mode = "normal"

        if test_mode == "logistics_reroute_failure":
            st.warning(
                "TEST ONLY — controlled recovery demonstration. CORTEX will simulate a transient "
                "fulfilment execution timeout after valid carrier planning. This is NOT a failure "
                "of orders.csv, products.csv, carriers.json or the carrier-selection business logic. "
                "The failed test agent is penalised, Recovery Controller classifies it as recoverable, "
                "and Adaptive Router reroutes the same work to a different compatible executor."
            )
        elif test_mode == "logistics_stop_failure":
            st.warning(
                "TEST ONLY — the Unrecoverable Failure Agent will fail. Recovery Controller will "
                "choose STOP, and Verification, TrustGraph, Governance, Quality and Report will not execute."
            )

        render_workflow_preview("logistics", test_mode)

    launch = st.button(
        "Launch Multi-Agent Mission",
        type="primary",
        use_container_width=True,
        disabled=not dataset_ready,
        help=(
            "Add the required source data first."
            if not dataset_ready
            else "Launch the detected CORTEX workflow."
        ),
    )
    if launch:
        payload: dict[str, Any] = {
            "test_mode": test_mode,
            "use_default_data": False,
        }
        if profile == "complaint":
            payload["complaint_document_id"] = st.session_state.complaint_document_id
        try:
            with st.spinner("Queuing CORTEX workflow..."):
                result = api_request(
                    "POST",
                    f"/api/v1/missions/{st.session_state.selected_mission_id}/execute",
                    payload=payload,
                )
            st.session_state.selected_run_id = result["workflow_run_id"]
            st.success("Workflow started. Open Workflow Monitor to follow live progress.")
        except RuntimeError as exc:
            st.error(str(exc))


def render_mission_plan() -> None:
    page_header("Mission Architecture", "Plan and agent allocation", "Inspect task dependencies, execution modes and routing rationale.")
    mission = mission_selector(load_missions(), "plan_mission_selector")
    if not mission:
        return
    try:
        detail = api_request("GET", f"/api/v1/missions/{mission['id']}")
    except RuntimeError as exc:
        st.error(str(exc))
        return
    profile = detect_ui_workflow_profile(detail)
    st.markdown(f"### {detail['title']} &nbsp; {badge(detail['status'])}", unsafe_allow_html=True)
    st.caption(detail["objective"])
    st.caption(
        "Detected workflow: Logistics Intelligence"
        if profile == "logistics"
        else "Detected workflow: Customer Complaint Intelligence"
    )

    tasks = detail.get("tasks", [])
    if profile == "complaint":
        if not any(item.get("task_type") == "adversarial_testing" for item in tasks):
            tasks = list(tasks) + [
                {
                    "sequence_order": 4,
                    "title": "Run continuous adversarial claim challenge",
                    "task_type": "adversarial_testing",
                    "execution_mode": "sequential",
                    "assigned_agent_name": "Adversarial Agent",
                    "assigned_agent_code": "adversarial_test_agent",
                    "risk_level": "low",
                    "status": "planned",
                }
            ]
        complaint_order = {
            "mission_planning": 1,
            "knowledge_retrieval": 2,
            "complaint_analysis": 3,
            "root_cause_analysis": 4,
            "adversarial_testing": 5,
            "claim_verification": 6,
            "governance_review": 7,
            "quality_evaluation": 8,
            "report_generation": 9,
        }
        tasks = sorted(
            tasks,
            key=lambda item: (
                complaint_order.get(str(item.get("task_type", "")), 99),
                int(item.get("sequence_order", 99) or 99),
            ),
        )
    if profile == "logistics":
        logistics_types = {
            "mission_planning", "logistics_routing", "logistics_execution", "workflow_recovery",
            "claim_verification", "governance_review", "quality_evaluation", "report_generation",
        }
        relevant = [item for item in tasks if item.get("task_type") in logistics_types]
        if relevant:
            tasks = relevant
        else:
            st.info(
                "The mission was created with the generic template. Launch it once and the live "
                "logistics planner will replace those placeholders with the domain-specific plan."
            )
            preview = pd.DataFrame(
                [
                    {"Stage": 1, "Task": "Define logistics mission scope", "Agent": "Mission Planner", "Status": "Planned"},
                    {"Stage": 2, "Task": "Optimise logistics routing", "Agent": "Logistics Routing Agent", "Status": "Planned"},
                    {"Stage": 3, "Task": "Execute simulated fulfilment", "Agent": "Task Execution / compatible fallback", "Status": "Planned"},
                    {"Stage": 4, "Task": "Verify fulfilment evidence", "Agent": "Verification Agent", "Status": "Planned"},
                    {"Stage": 5, "Task": "Apply governance", "Agent": "Guardian Governance Agent", "Status": "Planned"},
                    {"Stage": 6, "Task": "Evaluate quality", "Agent": "Quality Evaluator", "Status": "Planned"},
                    {"Stage": 7, "Task": "Generate report", "Agent": "Executive Report Agent", "Status": "Planned"},
                ]
            )
            st.dataframe(preview, use_container_width=True, hide_index=True)
            tasks = []

    if tasks:
        frame = pd.DataFrame(
            [
                {
                    "Stage": index,
                    "Task": item["title"],
                    "Mode": item["execution_mode"],
                    "Agent": (
                        "Adversarial Agent"
                        if str(item.get("assigned_agent_code", "")) == "adversarial_test_agent"
                        else item.get("assigned_agent_name") or "Pending"
                    ),
                    "Risk": item["risk_level"],
                    "Status": item["status"],
                }
                for index, item in enumerate(tasks, start=1)
            ]
        )
        st.dataframe(frame, use_container_width=True, hide_index=True)

    try:
        latest = api_request("GET", f"/api/v1/missions/{mission['id']}/workflow/latest")
        latest_state = latest.get("state_snapshot", {})
        routing = latest_state.get("routing", [])
        recovery_routing = latest_state.get("recovery_routing", [])
    except RuntimeError:
        latest = {}
        latest_state = {}
        routing = []
        recovery_routing = []
    if routing:
        st.subheader("Adaptive routing explanation")
        for item in routing:
            if str(item.get("task_type")) == "adversarial_testing":
                continue
            st.markdown(
                f'<div class="finding-card"><b>{html.escape(item["task_type"].replace("_"," ").title())}</b> → '
                f'<span style="color:var(--cyan)">{html.escape(item["selected_agent_name"])}</span><br>'
                f'<span class="metric-detail">Score {item["score"]:.1f} · {html.escape(item["reason"])}</span></div>',
                unsafe_allow_html=True,
            )
    if profile == "logistics" and latest_state:
        run_mode = str(latest_state.get("test_mode", "normal") or "normal")
        mode_name = {
            "normal": "Normal Logistics",
            "logistics_reroute_failure": "Recoverable Failure — Reroute",
            "logistics_stop_failure": "Unrecoverable Failure — Stop",
        }.get(run_mode, run_mode.replace("_", " ").title())
        st.subheader("Logistics recovery path")
        st.markdown(
            '<div class="finding-card"><div class="metric-label">Execution mode</div>'
            f'<div class="timeline-title">{html.escape(mode_name)}</div>'
            f'<div class="metric-detail">The same natural-language mission uses a different LangGraph recovery branch only when a failure-test mode is selected.</div></div>',
            unsafe_allow_html=True,
        )
        failure_context = latest_state.get("failure_context") or {}
        if isinstance(failure_context, dict) and failure_context:
            business_failure = "YES" if bool(failure_context.get("business_logic_failure")) else "NO"
            st.markdown(
                '<div class="finding-card" style="border-color:rgba(255,127,155,.52)">'
                '<div class="metric-label">Why did the execution fail?</div>'
                f'<div class="timeline-title">{html.escape(str(failure_context.get("classification", "failure")).upper())}</div>'
                f'<div class="metric-detail"><b>Origin:</b> {html.escape(str(failure_context.get("origin", "Execution failure")))}<br>'
                f'<b>Reason:</b> {html.escape(str(failure_context.get("reason", "")))}<br>'
                f'<b>Business-data / carrier-selection failure:</b> {business_failure}<br>'
                f'<b>Impact:</b> {html.escape(str(failure_context.get("impact", "")))}</div></div>',
                unsafe_allow_html=True,
            )

        failure_telemetry = latest_state.get("failed_agent_telemetry") or {}
        if isinstance(failure_telemetry, dict) and failure_telemetry:
            st.markdown(
                '<div class="finding-card" style="border-color:rgba(255,127,155,.52)">'
                '<div class="metric-label">Failed agent — persisted telemetry</div>'
                f'<div class="timeline-title"><span class="stage-status-icon failed"></span>{html.escape(str(failure_telemetry.get("agent_name") or failure_telemetry.get("agent_code") or "Failed agent"))}</div>'
                f'<div class="metric-detail">Trust {float(failure_telemetry.get("trust_before",0) or 0):.2f} → {float(failure_telemetry.get("trust_after",0) or 0):.2f} · '
                f'Reliability {float(failure_telemetry.get("reliability_before",0) or 0):.2f} → {float(failure_telemetry.get("reliability_after",0) or 0):.2f}</div></div>',
                unsafe_allow_html=True,
            )

        recovery = latest_state.get("recovery_result", {}).get("output", {}).get("recovery", {})
        recovery_context = latest_state.get("recovery_context") or {}
        if isinstance(recovery, dict) and recovery:
            action = str(recovery.get("action", "stop"))
            why = str(
                (recovery_context or {}).get("why")
                or recovery.get("reason", "")
            )
            router_instruction = str((recovery_context or {}).get("router_instruction") or "")
            detail = f'<b>Why:</b> {html.escape(why)}'
            if router_instruction:
                detail += f'<br><b>Recovery policy:</b> {html.escape(router_instruction)}'
            st.markdown(
                '<div class="finding-card" style="border-color:rgba(255,200,107,.42)">'
                '<div class="metric-label">Why did CORTEX choose this recovery action?</div>'
                f'<div class="timeline-title">{html.escape(action.upper())}</div>{badge(action)}'
                f'<div class="metric-detail" style="margin-top:.45rem">{detail}</div></div>',
                unsafe_allow_html=True,
            )
        if latest_state.get("stopped_reason"):
            st.error(
                "Workflow stopped safely. Verification, TrustGraph, Governance, Human Approval, Quality Evaluation and Executive Report were not executed."
            )

    if recovery_routing:
        st.subheader("Recovery rerouting")
        for item in recovery_routing:
            st.markdown(
                f'<div class="finding-card"><div class="metric-label">Same work, new agent</div>'
                f'<b>{html.escape(item["selected_agent_name"])}</b> {badge("rerouted")}<br>'
                f'<span class="metric-detail">Recovery score {item["score"]:.1f} · {html.escape(item["reason"])}</span></div>',
                unsafe_allow_html=True,
            )


def render_workflow_monitor() -> None:
    page_header("Live Orchestration", "Workflow Monitor", "Track the domain-specific agent path, failures, recovery decisions and verified output.")
    mission = mission_selector(load_missions(), "workflow_mission_selector")
    if not mission:
        return
    c1, _ = st.columns([1, 4])
    with c1:
        st.button("Refresh status", use_container_width=True)
    try:
        run = api_request("GET", f"/api/v1/missions/{mission['id']}/workflow/latest")
    except RuntimeError as exc:
        st.info(str(exc))
        return
    st.session_state.selected_run_id = run["id"]
    state = run.get("state_snapshot", {})
    profile = str(state.get("workflow_profile") or detect_ui_workflow_profile(mission)).lower()
    test_mode = str(state.get("test_mode", "normal") or "normal").lower()
    mode_labels = {
        "normal": "Normal Logistics" if profile == "logistics" else "Normal",
        "adversarial": "Continuous Assurance",
        "logistics_reroute_failure": "Recoverable Failure — Reroute",
        "logistics_stop_failure": "Unrecoverable Failure — Stop",
    }
    profile_label = "Logistics" if profile == "logistics" else "Customer Complaint"

    render_metric_grid(
        [
            ("Workflow status", run["status"].replace("_", " ").title(), "Current execution state"),
            ("Progress", f"{run['progress_percent']:.0f}%", "Mission completion"),
            ("Tokens", f"{run['total_tokens']:,}", "Azure OpenAI consumption"),
            ("Latency", f"{run['total_latency_ms']/1000:.1f}s", "End-to-end runtime"),
            ("Workflow", profile_label, "Detected from the mission objective"),
            ("Execution mode", mode_labels.get(test_mode, test_mode.replace("_", " ").title()), "Per-run control"),
        ],
        min_width=230,
    )
    st.progress(min(max(run["progress_percent"] / 100.0, 0.0), 1.0))
    st.markdown(badge(run["status"]), unsafe_allow_html=True)

    if profile == "complaint":
        pass
    elif test_mode == "logistics_reroute_failure":
        st.warning(
            "TEST ONLY — controlled recoverable failure. A transient fulfilment execution timeout is "
            "simulated after valid carrier planning; it is not a carrier-data or routing-business-logic failure."
        )
    elif test_mode == "logistics_stop_failure":
        st.error(
            "TEST ONLY — controlled unrecoverable failure. An execution-integrity fault is simulated "
            "after valid carrier planning; Recovery Controller must stop before unverified output proceeds."
        )

    timeline = run.get("timeline", [])
    stage_status = {item.get("stage"): item.get("status") for item in timeline}
    stopped = bool(state.get("stopped_reason")) and profile == "logistics"
    skipped_after_stop = {
        "verify_claims", "build_trust_graph", "apply_governance", "request_approval",
        "evaluate_quality", "generate_report",
    }

    cards = ['<div class="stage-grid">']
    for index, (stage, label) in enumerate(
        workflow_stage_labels(profile, test_mode, state),
        start=1,
    ):
        status_value = str(stage_status.get(stage, "waiting") or "waiting")
        if (
            profile == "complaint"
            and test_mode != "adversarial"
            and run.get("status") == "completed"
            and stage == "inject_adversarial_claim"
        ):
            status_value = "skipped"
        if stopped and stage in skipped_after_stop:
            status_value = "skipped"
        if (
            stage == "request_approval"
            and run.get("status") == "completed"
            and not state.get("approval_required")
            and stage not in stage_status
        ):
            status_value = "skipped"
        css = (
            "done" if status_value == "completed"
            else "failed" if status_value == "failed"
            else "skipped" if status_value == "skipped"
            else "active" if stage == run.get("current_stage")
            else ""
        )
        icon_status = (
            "active" if css == "active" and status_value == "waiting" else status_value
        )
        cards.append(
            f'<div class="stage-card {css}"><div class="stage-index">{index:02d}</div>'
            f'<div class="stage-name"><span class="stage-status-icon {html.escape(icon_status)}"></span>{html.escape(label)}</div>'
            f'{badge(status_value)}</div>'
        )
    cards.append("</div>")
    st.markdown("".join(cards), unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.2, .8], gap="large")
    with left:
        layout_marker("workflow-split")
        st.subheader("Execution timeline")
        if timeline:
            st.markdown('<div class="timeline">', unsafe_allow_html=True)
            for item in reversed(timeline[-18:]):
                status_value = str(item.get("status", "unknown") or "unknown").lower()
                timeline_message = str(item.get("message", ""))
                if str(item.get("stage", "")) == "inject_adversarial_claim" or str(item.get("agent_code", "")) == "adversarial_test_agent":
                    timeline_message = timeline_message.replace("Adversarial Test Agent", "Adversarial Agent")
                    timeline_message = timeline_message.replace("TEST ONLY:", "").replace("TEST ONLY", "")
                    timeline_message = timeline_message.replace("TEST-FAKE-001", "an unapproved evidence reference")
                    timeline_message = timeline_message.replace("test-only", "assurance").replace("Test-only", "Assurance")
                st.markdown(
                    f'<div class="timeline-item timeline-{html.escape(status_value)}"><div class="timeline-meta">'
                    f'<span>{html.escape(str(item.get("stage","workflow")).replace("_"," ").title())}</span>'
                    f'<span>{item.get("latency_ms",0)} ms · {item.get("tokens",0)} tokens</span></div>'
                    f'<div class="timeline-title">{html.escape(timeline_message.strip())}</div>'
                    f'{badge(status_value)}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.caption("The workflow is queued. Refresh after a few seconds.")

    with right:
        st.subheader("Mission intelligence")
        st.markdown(
            f'<div class="glass"><div class="metric-label">Current Stage</div>'
            f'<div class="metric-value" style="font-size:1.25rem">{html.escape(str(run.get("current_stage") or "Queued").replace("_"," ").title())}</div>'
            f'<div class="metric-detail">Retries / recovery reroutes: {run["retry_count"]}</div></div>',
            unsafe_allow_html=True,
        )
        if run["status"] == "awaiting_approval":
            st.warning("A high-impact recommendation is waiting for a reviewer.")
        if run["status"] == "completed":
            st.success("Verified report ready.")

        if profile == "logistics":
            plan_payload = state.get("logistics_plan_result", {}).get("output", {}).get("logistics_plan", {})
            if isinstance(plan_payload, dict) and plan_payload:
                st.markdown(
                    '<div class="finding-card"><div class="metric-label">Logistics plan</div>'
                    f'<div class="timeline-title">{int(plan_payload.get("orders_planned",0))} orders planned</div>'
                    f'<div class="metric-detail">Mode {html.escape(str(plan_payload.get("optimization_mode","balanced")))} · '
                    f'Cost {float(plan_payload.get("total_shipping_cost",0) or 0):.2f} · '
                    f'Average ETA {float(plan_payload.get("average_eta_days",0) or 0):.1f} days</div></div>',
                    unsafe_allow_html=True,
                )

            failure_context = state.get("failure_context") or {}
            if isinstance(failure_context, dict) and failure_context:
                classification = str(failure_context.get("classification", "unknown")).upper()
                business_logic = "YES" if failure_context.get("business_logic_failure") else "NO"
                st.markdown(
                    '<div class="finding-card" style="border-color:rgba(255,127,155,.5)">'
                    '<div class="metric-label">Why did the agent fail?</div>'
                    f'<div class="timeline-title">{html.escape(classification)} controlled test failure</div>'
                    f'<div class="metric-detail"><b>Origin:</b> {html.escape(str(failure_context.get("origin", "Controlled TEST ONLY execution-mode injection")))}<br>'
                    f'<b>Reason:</b> {html.escape(str(failure_context.get("reason", "")))}<br>'
                    f'<b>Business-logic/data failure:</b> {business_logic}<br>'
                    f'<b>Impact:</b> {html.escape(str(failure_context.get("impact", "")))}</div></div>',
                    unsafe_allow_html=True,
                )

            if state.get("failed_agent_code"):
                failure_telemetry = state.get("failed_agent_telemetry") or {}
                try:
                    agents = api_request("GET", "/api/v1/agents")
                    failed_agent = next(
                        (agent for agent in agents if agent.get("code") == state.get("failed_agent_code")),
                        None,
                    )
                except RuntimeError:
                    failed_agent = None

                agent_name = str(
                    failure_telemetry.get("agent_name")
                    or (failed_agent or {}).get("name")
                    or state.get("failed_agent_code")
                )
                if failure_telemetry:
                    telemetry_text = (
                        f'Trust {float(failure_telemetry.get("trust_before",0) or 0):.2f} → '
                        f'{float(failure_telemetry.get("trust_after",0) or 0):.2f} · Reliability '
                        f'{float(failure_telemetry.get("reliability_before",0) or 0):.2f} → '
                        f'{float(failure_telemetry.get("reliability_after",0) or 0):.2f}'
                    )
                elif failed_agent:
                    telemetry_text = (
                        f'Trust {float(failed_agent.get("trust_score",0) or 0):.2f} · Reliability '
                        f'{float(failed_agent.get("reliability_score",0) or 0):.2f}'
                    )
                else:
                    telemetry_text = "Failure telemetry was persisted to the audit trail."

                st.markdown(
                    '<div class="finding-card" style="border-color:rgba(255,127,155,.5)">'
                    '<div class="metric-label">Failed agent telemetry</div>'
                    f'<div class="timeline-title"><span class="stage-status-icon failed"></span>{html.escape(agent_name)}</div>'
                    f'<div class="metric-detail">{html.escape(telemetry_text)} · Failure penalty persisted to Agent Registry.</div></div>',
                    unsafe_allow_html=True,
                )

            recovery = state.get("recovery_result", {}).get("output", {}).get("recovery", {})
            recovery_context = state.get("recovery_context") or {}
            if isinstance(recovery, dict) and recovery:
                action = str(recovery.get("action", "stop"))
                why = str(
                    (recovery_context or {}).get("why")
                    or recovery.get("reason", "")
                )
                route_instruction = str((recovery_context or {}).get("router_instruction") or "")
                detail = why + (f" {route_instruction}" if route_instruction else "")
                st.markdown(
                    '<div class="finding-card" style="border-color:rgba(255,200,107,.42)">'
                    '<div class="metric-label">Why did CORTEX choose this recovery action?</div>'
                    f'<div class="timeline-title">Recovery Controller -> {html.escape(action.upper())}</div>{badge(action)}'
                    f'<div class="metric-detail" style="margin-top:.45rem">{html.escape(detail)}</div></div>',
                    unsafe_allow_html=True,
                )
            if state.get("recovery_routing"):
                route = state["recovery_routing"][0]
                st.success(
                    f"Same logistics work rerouted to {route.get('selected_agent_name', route.get('selected_agent_code'))}. "
                    "The original primary executor and failed TEST ONLY agent were excluded before Adaptive Router scored the fallback candidates."
                )
            if stopped:
                st.error(
                    "Workflow stopped safely. Verification, TrustGraph, Governance, Human Approval, Quality Evaluation and Executive Report were skipped."
                )
                st.caption(str(state.get("stopped_reason")))

        if profile == "complaint":
            verification = (
                state.get("verification_result", {})
                .get("output", {})
                .get("verification", {})
            )
            claims = verification.get("claims", []) if isinstance(verification, dict) else []
            if claims:
                st.markdown("#### Claim verification proof")
                proof_cards: list[str] = []
                for claim in claims:
                    if not isinstance(claim, dict):
                        continue
                    finding_id = str(claim.get("finding_id", "Claim"))
                    result_label = str(claim.get("verification_result", "unknown"))
                    source_code = str(claim.get("source_agent_code", "root_cause_analysis"))
                    invalid_ids = ", ".join(
                        str(value) for value in claim.get("invalid_evidence_ids", [])
                    ) or "None"
                    is_adversarial = source_code == "adversarial_test_agent"
                    border = (
                        "border-color:rgba(255,127,155,.58)"
                        if is_adversarial
                        else "border-color:rgba(115,243,189,.28)"
                    )
                    source_label = "Adversarial Agent" if is_adversarial else source_code
                    details = (
                        f"Source {source_label} · Groundedness "
                        f"{float(claim.get('groundedness', 0) or 0):.2f} · Hallucination risk "
                        f"{float(claim.get('hallucination_risk', 0) or 0):.2f}"
                    )
                    if is_adversarial:
                        details += " · Evidence status INVALID"
                    proof_cards.append(
                        f'<div class="finding-card" style="{border}">'
                        f'<div class="timeline-title">{html.escape(finding_id)} → {html.escape(result_label.upper())}</div>'
                        f'{badge(result_label)}'
                        f'<div class="metric-detail" style="margin-top:.45rem">{html.escape(details)}</div>'
                        '</div>'
                    )
                if proof_cards:
                    st.markdown("".join(proof_cards), unsafe_allow_html=True)

        if state.get("last_error") and not stopped:
            st.error(state["last_error"])


def render_agent_registry() -> None:
    page_header("Digital Workforce", "Agent Registry", "Operational passports for every specialist in the CORTEX workforce.")
    try:
        agents = api_request("GET", "/api/v1/agents")
    except RuntimeError as exc:
        st.error(str(exc)); return
    if not agents: st.info("No agents registered."); return
    df = pd.DataFrame(agents)
    operational_df = df[df["status"].astype(str).str.lower() == "active"]
    if operational_df.empty:
        operational_df = df
    render_metric_grid(
        [
            ("Registered", len(agents), "Includes controlled test agents"),
            ("Average trust", f"{operational_df['trust_score'].mean():.1f}", "Active agents only"),
            ("Average reliability", f"{operational_df['reliability_score'].mean():.1f}", "Active agents only"),
            ("Avg hallucination", f"{operational_df['hallucination_rate'].mean():.1f}%", "Active agents only; lower is better"),
        ],
        min_width=220,
    )

    selected_mission_id = st.session_state.get("selected_mission_id")
    if selected_mission_id:
        try:
            selected_run = api_request(
                "GET", f"/api/v1/missions/{selected_mission_id}/workflow/latest"
            )
            selected_state = selected_run.get("state_snapshot", {})
        except RuntimeError:
            selected_state = {}
        failure_telemetry = selected_state.get("failed_agent_telemetry") or {}
        if isinstance(failure_telemetry, dict) and failure_telemetry:
            st.markdown(
                '<div class="finding-card" style="border-color:rgba(255,127,155,.5)">'
                '<div class="metric-label">Latest selected mission — failure impact</div>'
                f'<div class="timeline-title"><span class="stage-status-icon failed"></span>{html.escape(str(failure_telemetry.get("agent_name") or failure_telemetry.get("agent_code") or "Failure agent"))}</div>'
                f'<div class="metric-detail">Trust {float(failure_telemetry.get("trust_before",0) or 0):.2f} → {float(failure_telemetry.get("trust_after",0) or 0):.2f} · '
                f'Reliability {float(failure_telemetry.get("reliability_before",0) or 0):.2f} → {float(failure_telemetry.get("reliability_after",0) or 0):.2f}. '
                'The values below come from the persisted Agent Registry.</div></div>',
                unsafe_allow_html=True,
            )

    # A trust-vs-reliability scatter becomes unreadable when many agents have nearly
    # identical scores. Render one horizontal row per agent instead: the connector
    # shows the gap between trust and reliability, while the two marker types make
    # the metrics immediately distinguishable. Hallucination rate remains available
    # in hover details instead of being encoded as a confusing colour scale.
    registry_plot = df.copy()
    registry_plot["display_name"] = registry_plot.apply(
        lambda row: (
            "Adversarial Agent"
            if str(row.get("code", "")) == "adversarial_test_agent"
            else str(row.get("name", row.get("code", "Agent")))
        ),
        axis=1,
    )
    registry_plot["trust_score"] = pd.to_numeric(registry_plot["trust_score"], errors="coerce").fillna(0.0)
    registry_plot["reliability_score"] = pd.to_numeric(registry_plot["reliability_score"], errors="coerce").fillna(0.0)
    registry_plot["hallucination_rate"] = pd.to_numeric(registry_plot["hallucination_rate"], errors="coerce").fillna(0.0)
    registry_plot["completed_runs"] = pd.to_numeric(registry_plot["completed_runs"], errors="coerce").fillna(0).astype(int)
    registry_plot["agent_score"] = (registry_plot["trust_score"] + registry_plot["reliability_score"]) / 2.0
    registry_plot = registry_plot.sort_values(
        ["agent_score", "display_name"], ascending=[False, True]
    ).reset_index(drop=True)

    connector_x: list[float | None] = []
    connector_y: list[str | None] = []
    for _, row in registry_plot.iterrows():
        connector_x.extend([float(row["trust_score"]), float(row["reliability_score"]), None])
        connector_y.extend([str(row["display_name"]), str(row["display_name"]), None])

    hover_data = registry_plot[[
        "display_name", "trust_score", "reliability_score",
        "hallucination_rate", "completed_runs", "status"
    ]].to_numpy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=connector_x,
            y=connector_y,
            mode="lines",
            line=dict(width=2, color="rgba(145,160,187,.35)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=registry_plot["trust_score"],
            y=registry_plot["display_name"],
            mode="markers",
            name="Trust score",
            customdata=hover_data,
            marker=dict(size=13, symbol="circle", color="#65e7ff", line=dict(width=1, color="#dff9ff")),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Trust: %{customdata[1]:.1f}<br>"
                "Reliability: %{customdata[2]:.1f}<br>"
                "Hallucination: %{customdata[3]:.1f}%<br>"
                "Completed runs: %{customdata[4]}<br>"
                "Status: %{customdata[5]}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=registry_plot["reliability_score"],
            y=registry_plot["display_name"],
            mode="markers",
            name="Reliability score",
            customdata=hover_data,
            marker=dict(size=13, symbol="diamond", color="#a86dff", line=dict(width=1, color="#efe3ff")),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Trust: %{customdata[1]:.1f}<br>"
                "Reliability: %{customdata[2]:.1f}<br>"
                "Hallucination: %{customdata[3]:.1f}%<br>"
                "Completed runs: %{customdata[4]}<br>"
                "Status: %{customdata[5]}<extra></extra>"
            ),
        )
    )

    minimum_score = float(
        min(registry_plot["trust_score"].min(), registry_plot["reliability_score"].min())
    )
    x_min = max(0.0, minimum_score - 3.0)
    chart_height = max(560, 36 * len(registry_plot) + 120)
    fig.update_layout(
        template="plotly_dark",
        height=chart_height,
        margin=dict(l=225, r=30, t=62, b=55),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,16,32,.55)",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(
            title="Score",
            range=[x_min, 100.8],
            dtick=5,
            gridcolor="rgba(151,178,255,.14)",
            zeroline=False,
        ),
        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=registry_plot["display_name"].tolist(),
            autorange="reversed",
            automargin=True,
            tickfont=dict(size=12),
            gridcolor="rgba(0,0,0,0)",
        ),
    )
    fig.add_vline(
        x=90,
        line_width=1,
        line_dash="dot",
        line_color="rgba(115,243,189,.45)",
        annotation_text="90 benchmark",
        annotation_position="top",
    )

    st.markdown("### Trust & reliability by agent")
    st.caption(
        "Each row is one agent. Circle = trust score; diamond = reliability score. "
        "The connecting line shows the gap between the two metrics. Hallucination rate "
        "is available on hover instead of using a separate colour scale."
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": False},
    )

    cards: list[str] = []
    for agent in agents:
        agent_code = str(agent.get("code", ""))
        capabilities = (
            "hypothesis challenge · evidence quality analysis · verification challenge"
            if agent_code == "adversarial_test_agent"
            else " · ".join(agent.get("capabilities", [])[:3])
        )
        lane_chips: list[str] = []
        if agent_code == "adversarial_test_agent":
            lane_chips.append('<span class="chip chip-unknown">Complaint assurance</span>')
        if agent_code in LOGISTICS_AGENT_CODES:
            lane_chips.append('<span class="chip chip-unknown">Logistics lane</span>')
        if agent_code == "recoverable_failure_agent":
            lane_chips.append('<span class="chip chip-unknown">Reroute test</span>')
        elif agent_code == "unrecoverable_failure_agent":
            lane_chips.append('<span class="chip chip-unknown">Stop test</span>')
        elif agent_code == "recovery_controller":
            lane_chips.append('<span class="chip chip-unknown">Recovery decision</span>')
        elif agent_code == "logistics_backup_execution":
            lane_chips.append('<span class="chip chip-unknown">Fallback execution</span>')
        lane_chip = "".join(lane_chips)
        display_name = (
            "Adversarial Agent" if agent_code == "adversarial_test_agent" else agent["name"]
        )
        display_description = (
            "Challenges causal findings with competing hypotheses so evidence quality, groundedness and verification controls are continuously exercised."
            if agent_code == "adversarial_test_agent"
            else agent["description"]
        )
        status_badge = "" if agent_code == "adversarial_test_agent" else badge(agent["status"])
        cards.append(
            '<article class="agent-card">'
            f'<div class="eyebrow">{html.escape(agent["code"])}</div>'
            f'<h3>{html.escape(display_name)}</h3>'
            f'<div class="agent-description">{html.escape(display_description)}</div>'
            '<div class="divider"></div>'
            '<div class="metric-label">Capabilities</div>'
            f'<div class="agent-capabilities">{html.escape(capabilities)}</div>'
            '<div class="agent-badges">'
            f'{status_badge}'
            f'{lane_chip}'
            f'<span class="chip chip-ready">Trust {agent["trust_score"]:.0f}</span>'
            f'<span class="chip chip-ready">Reliable {agent["reliability_score"]:.0f}</span>'
            f'<span class="chip chip-unknown">Hallucination {agent["hallucination_rate"]:.1f}%</span>'
            f'<span class="chip chip-unknown">Runs {agent["completed_runs"]}</span>'
            '</div></article>'
        )
    st.markdown(f'<div class="agent-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_knowledge_base() -> None:
    page_header(
        "Enterprise RAG",
        "Knowledge Base",
        "Index approved policies and retrieve evidence-grounded answers with citations.",
    )
    try:
        status = api_request("GET", "/api/v1/knowledge/status")
        st.markdown(
            f'{badge(status["status"])} &nbsp; '
            f'<span class="metric-detail">{status["document_chunks"]} indexed chunks · '
            f'{html.escape(status["collection"])}</span>',
            unsafe_allow_html=True,
        )
    except RuntimeError as exc:
        st.error(str(exc))
        return

    ask_tab, upload_tab = st.tabs(["Ask knowledge base", "Add approved knowledge"])

    with ask_tab:
        st.subheader("Ask the enterprise knowledge base")
        query = st.text_input(
            "Question",
            value="When must a customer complaint be escalated and which actions need human approval?",
        )
        if st.button("Retrieve grounded answer", use_container_width=True):
            try:
                with st.spinner("Searching approved enterprise evidence..."):
                    result = api_request(
                        "GET",
                        "/api/v1/knowledge/search",
                        params={"q": query, "limit": 5},
                    )
                st.markdown(
                    '<div class="glass">'
                    + html.escape(result["answer"]).replace("\n", "<br>")
                    + '</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Confidence {result['confidence']:.2f} · {result['tokens']} tokens · "
                    f"{result['latency_ms']} ms"
                )
                with st.expander("Evidence sources"):
                    for item in result["evidence"]:
                        st.markdown(f"**{item['title']}** · score {item['relevance_score']:.3f}")
                        st.caption(item["text"][:420])
            except RuntimeError as exc:
                st.error(str(exc))

    with upload_tab:
        st.subheader("Add approved knowledge")
        document = st.file_uploader(
            "Markdown or text policy",
            type=["md", "txt"],
            key="knowledge_upload",
        )
        if document and st.button("Index knowledge", use_container_width=True):
            try:
                result = api_upload("/api/v1/uploads/knowledge", document)
                st.success(result["message"])
            except RuntimeError as exc:
                st.error(str(exc))


def render_approvals() -> None:
    page_header("Human Oversight", "Approval Centre", "Review high-impact actions before CORTEX allows them into a final decision.")
    if st.session_state.user["role"] not in {"reviewer", "administrator"}:
        st.warning("Reviewer access is required."); return
    try:
        approvals = api_request("GET", "/api/v1/approvals/pending")
    except RuntimeError as exc:
        st.error(str(exc)); return
    if not approvals:
        st.success("No pending approval requests."); return
    for item in approvals:
        st.markdown(
            f'<div class="glass"><div style="display:flex;justify-content:space-between"><div><div class="eyebrow">{html.escape(item.get("mission_title") or "Mission")}</div><h3>High-impact decision review</h3></div>{badge(item["risk_level"])}</div><p>{html.escape(item["reason"])}</p></div>',
            unsafe_allow_html=True,
        )
        actions = item.get("action_payload", {}).get("actions", [])
        if actions:
            st.dataframe(pd.DataFrame({"Proposed action": actions}), use_container_width=True, hide_index=True)
        comment = st.text_area("Reviewer comment", key=f"comment_{item['id']}", placeholder="Document the decision rationale.")
        c1, c2, c3 = st.columns(3)
        for column, action, label in [(c1,"approve","Approve"),(c2,"request_revision","Request Revision"),(c3,"reject","Reject")]:
            if column.button(label, key=f"{action}_{item['id']}", use_container_width=True):
                try:
                    api_request("POST", f"/api/v1/approvals/{item['id']}/decision", payload={"action": action, "comment": comment})
                    st.success(f"Decision recorded: {label}.")
                    st.rerun()
                except RuntimeError as exc:
                    st.error(str(exc))


def trust_graph_figure(graph: dict[str, Any]) -> go.Figure:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_ids = [node["node_id"] for node in nodes]
    positions: dict[str, tuple[float,float]] = {}
    groups: dict[str,list[dict]] = {}
    for node in nodes: groups.setdefault(node.get("node_type","other"), []).append(node)
    xmap = {"mission":0, "agent":1, "claim":2, "evidence":3, "invalid_evidence":3}
    for group, items in groups.items():
        x = xmap.get(group, 4)
        count = len(items)
        for index, node in enumerate(items):
            y = (index - (count-1)/2) * 1.25
            positions[node["node_id"]] = (x, y)
    edge_x, edge_y = [], []
    for edge in edges:
        if edge["source"] not in positions or edge["target"] not in positions: continue
        x0,y0=positions[edge["source"]]; x1,y1=positions[edge["target"]]
        edge_x += [x0,x1,None]; edge_y += [y0,y1,None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x,y=edge_y,mode="lines",line=dict(width=1,color="rgba(137,160,210,.35)"),hoverinfo="none"))
    colors={"mission":"#65e7ff","agent":"#7c8cff","claim":"#b47cff","evidence":"#73f3bd","invalid_evidence":"#ff7f9b"}
    for group, items in groups.items():
        fig.add_trace(go.Scatter(
            x=[positions[item["node_id"]][0] for item in items], y=[positions[item["node_id"]][1] for item in items],
            mode="markers", name=group.title(), text=[item.get("label",item["node_id"]) for item in items],
            hovertemplate="%{text}<extra></extra>", marker=dict(size=19 if group=="mission" else 14, color=colors.get(group,"#c4d0e8"), line=dict(width=1,color="white")),
        ))
    fig.update_layout(template="plotly_dark", height=max(520, len(nodes)*24), margin=dict(l=10,r=10,t=30,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,16,32,.48)", xaxis=dict(visible=False), yaxis=dict(visible=False), legend=dict(orientation="h"))
    return fig


def render_trust_graph() -> None:
    page_header("Decision Lineage", "TrustGraph", "Trace each material conclusion from agent output to supporting enterprise evidence.")
    mission = mission_selector(load_missions(), "trust_mission_selector")
    if not mission: return
    try:
        run = api_request("GET", f"/api/v1/missions/{mission['id']}/workflow/latest")
    except RuntimeError as exc:
        st.info(str(exc)); return
    state = run.get("state_snapshot", {})
    profile = str(state.get("workflow_profile") or detect_ui_workflow_profile(mission)).lower()
    if profile == "logistics" and state.get("stopped_reason"):
        st.error("TrustGraph was not generated because CORTEX stopped the logistics workflow before Verification.")
        st.caption(str(state.get("stopped_reason")))
        st.markdown(
            '<div class="finding-card" style="border-color:rgba(255,127,155,.5)">'
            '<div class="metric-label">Safe termination</div>'
            '<div class="timeline-title">Verification and evidence lineage were not executed</div>'
            '<div class="metric-detail">This is intentional: a stopped workflow must not create downstream assurance artifacts.</div></div>',
            unsafe_allow_html=True,
        )
        return
    try:
        graph = api_request("GET", f"/api/v1/workflow-runs/{run['id']}/trust-graph")
    except RuntimeError as exc:
        st.info(str(exc)); return
    summary = graph.get("summary", {})
    render_metric_grid(
        [
            ("Claims", summary.get("claim_count", 0), "Material conclusions"),
            ("Evidence", summary.get("evidence_count", 0), "Linked evidence items"),
            ("Verified", summary.get("verified_claims", 0), "Evidence-supported claims"),
            ("Review", summary.get("review_required", 0), "Human review required"),
            ("Rejected", summary.get("rejected_claims", 0), "Unsupported claims"),
            ("Invalid refs", summary.get("invalid_evidence_references", 0), "Unapproved evidence references"),
        ],
        min_width=165,
    )

    table = graph.get("claim_evidence_table", [])
    has_adversarial_claim = any(
        isinstance(row, dict)
        and str(row.get("source_agent_code", "")) == "adversarial_test_agent"
        for row in table
    )
    if has_adversarial_claim:
        st.markdown("### Hallucination defense proof")
        st.caption(
            "CORTEX evaluates every material claim against approved evidence. Unsupported claims remain visible in the lineage and are rejected before governance and reporting."
        )
        proof_cards: list[str] = []
        for row in table:
            if not isinstance(row, dict):
                continue
            finding_id = str(row.get("finding_id") or row.get("claim_id") or "Claim")
            status_value = str(row.get("verification_result", "unknown"))
            source = str(row.get("source_agent_code", "root_cause_analysis"))
            is_adversarial = source == "adversarial_test_agent"
            source_label = "Adversarial Agent" if is_adversarial else source.replace('_', ' ').title()
            invalid_ids = [str(value) for value in row.get("invalid_evidence_ids", [])]
            details = (
                f"Source: {source_label} · "
                f"Groundedness {float(row.get('groundedness', 0) or 0)*100:.0f}% · "
                f"Hallucination risk {float(row.get('hallucination_risk', 0) or 0)*100:.0f}%"
            )
            if invalid_ids:
                details += " · Evidence status: INVALID"
            proof_cards.append(
                '<div class="finding-card">'
                f'<div class="timeline-title">{html.escape(finding_id)} -> {html.escape(status_value.upper())}</div>'
                f'<div style="margin:.3rem 0 .5rem">{badge(status_value)}</div>'
                f'<div>{html.escape(str(row.get("claim", "")))}</div>'
                f'<div class="metric-detail" style="margin-top:.45rem">{html.escape(details)}</div>'
                '</div>'
            )
        if proof_cards:
            st.markdown("".join(proof_cards), unsafe_allow_html=True)

    display_graph = dict(graph)
    display_nodes = []
    for node in graph.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        item = dict(node)
        if str(item.get("node_id", "")) == "AGENT-adversarial_test_agent":
            item["label"] = "Adversarial Agent"
            item["status"] = "completed"
        if item.get("node_type") == "invalid_evidence" and "TEST-FAKE-001" in str(item.get("label", "")):
            item["label"] = "Rejected evidence reference"
        display_nodes.append(item)
    display_graph["nodes"] = display_nodes

    display_table = []
    for row in table:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        if str(item.get("source_agent_code", "")) == "adversarial_test_agent":
            item["source_agent_code"] = "adversarial_agent"
            if item.get("invalid_evidence_ids"):
                item["invalid_evidence_ids"] = ["INVALID"]
            gaps = []
            for value in item.get("evidence_gaps", []) or []:
                text = str(value).replace("TEST-FAKE-001", "an unapproved evidence reference")
                text = text.replace("controlled test", "unsupported")
                gaps.append(text)
            item["evidence_gaps"] = gaps
        display_table.append(item)

    st.plotly_chart(trust_graph_figure(display_graph), use_container_width=True)
    st.dataframe(pd.DataFrame(display_table), use_container_width=True, hide_index=True)



def render_shadowbench() -> None:
    page_header(
        "Safe Model Evaluation",
        "ShadowBench & Cost Analytics",
        "Replay an approved reporting task against a shadow deployment without affecting the customer-facing result.",
    )
    if st.session_state.user["role"] not in {"reviewer", "administrator"}:
        st.warning("Reviewer or administrator access is required.")
        return
    mission = mission_selector(load_missions(), "shadow_mission_selector")
    if not mission:
        return
    try:
        run = api_request("GET", f"/api/v1/missions/{mission['id']}/workflow/latest")
    except RuntimeError as exc:
        st.info(str(exc))
        return
    if run.get("status") != "completed":
        st.info("Complete the workflow before starting a shadow replay.")
        return
    left, right = st.columns([1.4, .6], gap="large")
    with left:
        layout_marker("shadow-split")
        st.markdown(
            '<div class="glass"><div class="metric-label">Replay Safety</div>'
            '<div class="metric-value" style="font-size:1.35rem">Isolated evaluation lane</div>'
            '<div class="metric-detail">Shadow output is stored for comparison and never replaces the approved mission report.</div></div>',
            unsafe_allow_html=True,
        )
    with right:
        if st.button("Run Shadow Replay", type="primary", use_container_width=True):
            try:
                with st.spinner("Running isolated candidate evaluation..."):
                    api_request(
                        "POST",
                        f"/api/v1/shadowbench/workflow-runs/{run['id']}/replay",
                        timeout=180,
                    )
                st.success("Shadow replay completed.")
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))
    try:
        records = api_request(
            "GET", f"/api/v1/shadowbench/workflow-runs/{run['id']}"
        )
    except RuntimeError as exc:
        st.error(str(exc))
        return
    if not records:
        st.info("No ShadowBench replay has been run for this mission.")
        return
    latest = records[0]
    prod = latest.get("production_metrics", {})
    shadow = latest.get("shadow_metrics", {})
    st.markdown("### Production vs shadow")
    frame = pd.DataFrame(
        [
            {"Metric": "Quality", "Production": prod.get("quality", 0), "Shadow": shadow.get("quality", 0)},
            {"Metric": "Groundedness", "Production": prod.get("groundedness", 0), "Shadow": shadow.get("groundedness", 0)},
            {"Metric": "Hallucination Risk", "Production": prod.get("hallucination_risk", 0), "Shadow": shadow.get("hallucination_risk", 0)},
            {"Metric": "Latency (ms)", "Production": prod.get("latency_ms", 0), "Shadow": shadow.get("latency_ms", 0)},
            {"Metric": "Tokens", "Production": prod.get("total_tokens", 0), "Shadow": shadow.get("total_tokens", 0)},
            {"Metric": "Estimated Cost", "Production": prod.get("estimated_cost", 0), "Shadow": shadow.get("estimated_cost", 0)},
        ]
    )
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.info(latest.get("recommendation", ""))
    st.caption(
        f"Production: {latest.get('production_deployment')} · Shadow: {latest.get('shadow_deployment')}"
    )

def render_report() -> None:
    page_header("Verified Output", "Executive Decision Report", "Only verified and governance-permitted findings appear in this report.")
    mission = mission_selector(load_missions(), "report_mission_selector")
    if not mission: return
    try:
        run = api_request("GET", f"/api/v1/missions/{mission['id']}/workflow/latest")
    except RuntimeError as exc:
        st.info(str(exc)); return
    state = run.get("state_snapshot", {})
    profile = str(state.get("workflow_profile") or detect_ui_workflow_profile(mission)).lower()
    if profile == "logistics" and state.get("stopped_reason"):
        st.error("No executive report was generated because CORTEX stopped the logistics workflow safely.")
        st.caption(str(state.get("stopped_reason")))
        st.markdown(
            '<div class="finding-card" style="border-color:rgba(255,127,155,.5)">'
            '<div class="metric-label">Final mission status</div>'
            '<div class="timeline-title">STOPPED - no executive decision report</div>'
            '<div class="metric-detail">Verification, TrustGraph, Governance, Human Approval, Quality Evaluation and Executive Report were skipped by the LangGraph stop branch.</div></div>',
            unsafe_allow_html=True,
        )
        if PDF_REPORTS_AVAILABLE and build_failure_audit_pdf is not None:
            try:
                audit_pdf = build_failure_audit_pdf(
                    mission=mission,
                    run=run,
                    state=state,
                )
                st.download_button(
                    "Download Failure & Recovery Audit PDF",
                    data=audit_pdf,
                    file_name=f"cortex_failure_recovery_audit_{mission['id']}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"Failure audit PDF could not be generated: {exc}")
        else:
            st.warning(
                "PDF export is temporarily unavailable because ReportLab is not installed "
                "in the running frontend environment. Rebuild the frontend image after "
                "updating requirements.final.txt."
            )
        return
    try:
        payload = api_request("GET", f"/api/v1/workflow-runs/{run['id']}/report")
    except RuntimeError as exc:
        st.info(str(exc)); return
    report = payload.get("output", {}).get("report", payload.get("report", payload))
    if profile == "logistics" and state.get("recovery_routing"):
        route = state.get("recovery_routing", [])[0]
        st.success(
            "Recovery completed — the failed logistics work was rerouted to "
            f"{route.get('selected_agent_name', route.get('selected_agent_code', 'a compatible fallback agent'))} "
            "and the verified workflow continued to reporting."
        )

    verification = (
        state.get("verification_result", {})
        .get("output", {})
        .get("verification", {})
    )
    claims = verification.get("claims", []) if isinstance(verification, dict) else []
    adversarial_claim = next(
        (
            item
            for item in claims
            if isinstance(item, dict)
            and item.get("source_agent_code") == "adversarial_test_agent"
        ),
        None,
    )
    if adversarial_claim and adversarial_claim.get("verification_result") == "rejected":
        st.success(
            "Adversarial assurance passed — RC-003 was rejected by Verification and excluded from the executive report."
        )
    score=float(report.get("mission_success_score",0))
    left,right=st.columns([1.6,.4],gap="large")
    with left:
        layout_marker("report-split")
        st.markdown(f'<div class="eyebrow">{html.escape(report.get("synthetic_data_notice",""))}</div><h1>{html.escape(report.get("title","CORTEX Report"))}</h1>',unsafe_allow_html=True)
        st.markdown('<div class="glass">'+html.escape(report.get("executive_summary","")).replace("\n","<br>")+'</div>',unsafe_allow_html=True)
    with right:
        st.markdown(f'<div class="score-ring" style="--score:{min(max(score,0),100)}%"><div class="score-text">{score:.1f}</div></div><div style="text-align:center;margin-top:.75rem">{badge(report.get("mission_status","Unknown"))}</div>',unsafe_allow_html=True)
    st.markdown('<div class="report-section"><h3>Verified Findings</h3></div>',unsafe_allow_html=True)
    for finding in report.get("key_findings",[]):
        st.markdown(f'<div class="finding-card"><div style="display:flex;justify-content:space-between;gap:1rem"><b>{html.escape(finding.get("category","Finding"))}</b>{badge(finding.get("verification_status","verified"))}</div><p>{html.escape(finding.get("finding",""))}</p><div class="metric-detail">Groundedness {finding.get("groundedness",0):.2f} · Confidence {finding.get("confidence",0):.2f} · Evidence {", ".join(finding.get("evidence_ids",[]))}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="report-section"><h3>Prioritised Recommendations</h3></div>',unsafe_allow_html=True)
    recs=report.get("recommendations",[])
    if recs:
        st.dataframe(pd.DataFrame(recs), use_container_width=True, hide_index=True)
    st.markdown('<div class="report-section"><h3>Governance & Next Action</h3></div>',unsafe_allow_html=True)
    st.info(report.get("governance_status",""))
    st.success(report.get("next_recommended_action",""))
    with st.expander("Risks and limitations"):
        for value in report.get("risks",[]): st.write("•",value)
        for value in report.get("limitations",[]): st.write("•",value)
    st.markdown('<div class="report-section"><h3>Downloadable Artifacts</h3></div>', unsafe_allow_html=True)
    report_pdf = None
    if PDF_REPORTS_AVAILABLE and build_executive_report_pdf is not None:
        try:
            report_pdf = build_executive_report_pdf(
                mission=mission,
                run=run,
                report=report,
                state=state,
            )
        except Exception as exc:
            st.error(f"Executive PDF could not be generated: {exc}")
    else:
        st.warning(
            "PDF export is temporarily unavailable because ReportLab is not installed "
            "in the running frontend environment. JSON export remains available. "
            "Rebuild the frontend image after updating requirements.final.txt."
        )

    download_json, download_pdf = st.columns(2)
    with download_json:
        st.download_button(
            "Download Executive Report JSON",
            data=json.dumps(report, indent=2, ensure_ascii=False, default=str),
            file_name=f"cortex_report_{mission['id']}.json",
            mime="application/json",
            use_container_width=True,
        )
    with download_pdf:
        if report_pdf is not None:
            st.download_button(
                "Download Executive Report PDF",
                data=report_pdf,
                file_name=f"cortex_executive_report_{mission['id']}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


def render_audit_logs() -> None:
    page_header("Operational Transparency", "Audit Logs", "A complete record of mission, agent, policy and approval activity.")
    mission_id = st.session_state.get("selected_mission_id")
    params={"limit":250}
    if mission_id and st.toggle("Filter to selected mission", value=False): params["mission_id"]=mission_id
    try:
        logs=api_request("GET","/api/v1/dashboard/audit-logs",params=params)
    except RuntimeError as exc:
        st.error(str(exc)); return
    if not logs: st.info("No audit activity yet."); return
    frame=pd.DataFrame(logs)
    columns=[c for c in ["timestamp","action","mission_id","entity_type","entity_id","details"] if c in frame]
    st.dataframe(frame[columns],use_container_width=True,hide_index=True,height=580)


def render_system_status() -> None:
    page_header("Platform Operations", "System Status", "Readiness across database, Azure OpenAI, ChromaDB and LangGraph persistence.")
    try:
        result=api_request("GET","/api/v1/system/components",timeout=90)
    except RuntimeError as exc:
        st.error(str(exc)); return
    st.markdown(badge(result["status"]),unsafe_allow_html=True)
    components=[("PostgreSQL",result["database"]),("Azure OpenAI",result["llm"]),("ChromaDB",result["vector_store"]),("LangGraph",result["workflow"])]
    cards: list[str] = []
    for name, value in components:
        details: list[str] = []
        for key, raw_value in value.items():
            if key in {"status", "usage"} or isinstance(raw_value, (dict, list)):
                continue
            display_value = str(raw_value)
            details.append(
                '<div class="component-row">'
                f'<span class="component-key">{html.escape(str(key).replace("_", " "))}</span>'
                f'<span class="component-value">{html.escape(display_value)}</span>'
                '</div>'
            )
        cards.append(
            '<article class="component-card">'
            f'<div class="metric-label">{html.escape(name)}</div>'
            f'<div class="component-title">{html.escape(str(value.get("status", "unknown")).title())}</div>'
            f'{badge(value.get("status", "unknown"))}'
            f'<div class="component-details">{"".join(details[:5])}</div>'
            '</article>'
        )
    st.markdown(f'<div class="component-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_application() -> None:
    page=render_sidebar()
    if page=="Mission Control": render_dashboard()
    elif page=="Create Mission": render_create_mission()
    elif page=="Mission Plan": render_mission_plan()
    elif page=="Workflow Monitor": render_workflow_monitor()
    elif page=="Agent Registry": render_agent_registry()
    elif page=="Knowledge Base": render_knowledge_base()
    elif page=="Approval Centre": render_approvals()
    elif page=="TrustGraph": render_trust_graph()
    elif page=="ShadowBench": render_shadowbench()
    elif page=="Executive Report": render_report()
    elif page=="Audit Logs": render_audit_logs()
    elif page=="System Status": render_system_status()


apply_design_system()
init_state()
if st.session_state.access_token and st.session_state.user:
    render_application()
else:
    render_login()
