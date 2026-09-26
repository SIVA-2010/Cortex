from __future__ import annotations

from io import BytesIO
from typing import Any, Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from xml.sax.saxutils import escape


NAVY = colors.HexColor("#0B1733")
CYAN = colors.HexColor("#1B9AAA")
GREEN = colors.HexColor("#1F8A70")
AMBER = colors.HexColor("#B7791F")
RED = colors.HexColor("#C2415D")
INK = colors.HexColor("#1C2638")
MUTED = colors.HexColor("#607089")
LINE = colors.HexColor("#D8E1EE")
PALE = colors.HexColor("#F4F7FB")
PALE_CYAN = colors.HexColor("#EAF8FA")
PALE_RED = colors.HexColor("#FDEFF2")
PALE_AMBER = colors.HexColor("#FFF7E8")
WHITE = colors.white


MODE_LABELS = {
    "normal": "Normal Logistics",
    "adversarial": "Continuous Assurance",
    "logistics_reroute_failure": "Recoverable Failure - Reroute",
    "logistics_stop_failure": "Unrecoverable Failure - Stop",
}


def _ascii(value: Any) -> str:
    """Normalize UI punctuation to glyphs supported by ReportLab base fonts."""
    text = str(value if value is not None else "")
    replacements = {
        "\u2192": "->",
        "\u2014": "-",
        "\u2013": "-",
        "\u2022": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _p(value: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(_ascii(value)).replace("\n", "<br/>"), style)


def _bullet(value: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph("&bull;&nbsp;" + escape(_ascii(value)), style)


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "CortexTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "CortexSubtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=MUTED,
            spaceAfter=14,
        ),
        "h2": ParagraphStyle(
            "CortexH2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13.5,
            leading=17,
            textColor=NAVY,
            spaceBefore=12,
            spaceAfter=7,
        ),
        "h3": ParagraphStyle(
            "CortexH3",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=14,
            textColor=CYAN,
            spaceBefore=8,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "CortexBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=13.5,
            textColor=INK,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "CortexSmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.7,
            leading=10.5,
            textColor=INK,
        ),
        "small_muted": ParagraphStyle(
            "CortexSmallMuted",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10.2,
            textColor=MUTED,
        ),
        "meta_label": ParagraphStyle(
            "CortexMetaLabel",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=11,
            textColor=MUTED,
        ),
        "meta_value": ParagraphStyle(
            "CortexMetaValue",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=INK,
        ),
        "callout": ParagraphStyle(
            "CortexCallout",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.6,
            leading=12.5,
            textColor=INK,
        ),
        "center": ParagraphStyle(
            "CortexCenter",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=INK,
            alignment=TA_CENTER,
        ),
        "table_header": ParagraphStyle(
            "CortexTableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=WHITE,
            alignment=TA_CENTER,
        ),
    }


def _footer(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 14 * mm, width - 18 * mm, 14 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 9 * mm, "CORTEX - Enterprise AgentOps Mission Control")
    canvas.drawRightString(width - 18 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _section_title(text: str, styles: dict[str, ParagraphStyle]) -> list[Any]:
    return [Spacer(1, 3), _p(text, styles["h2"])]


def _callout(
    title: str,
    body: str,
    styles: dict[str, ParagraphStyle],
    *,
    background: colors.Color = PALE_CYAN,
    accent: colors.Color = CYAN,
) -> Table:
    data = [[_p(title, styles["meta_value"])], [_p(body, styles["callout"])]]
    table = Table(data, colWidths=[174 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("LINEBEFORE", (0, 0), (0, -1), 3, accent),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _metadata_table(
    rows: Iterable[tuple[str, Any]], styles: dict[str, ParagraphStyle]
) -> Table:
    data = [
        [_p(label, styles["meta_label"]), _p(value, styles["meta_value"])]
        for label, value in rows
    ]
    table = Table(data, colWidths=[40 * mm, 134 * mm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _table(
    headers: list[str],
    rows: list[list[Any]],
    widths_mm: list[float],
    styles: dict[str, ParagraphStyle],
) -> Table:
    data: list[list[Any]] = [
        [_p(header, styles["table_header"]) for header in headers]
    ]
    for row in rows:
        data.append([_p(value, styles["small"]) for value in row])
    table = Table(
        data,
        colWidths=[value * mm for value in widths_mm],
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _mode_label(state: dict[str, Any], profile: str) -> str:
    mode = str(state.get("test_mode", "normal") or "normal")
    if mode == "normal" and profile != "logistics":
        return "Continuous Assurance"
    return MODE_LABELS.get(mode, _ascii(mode).replace("_", " ").title())


def _verification_claims(state: dict[str, Any]) -> list[dict[str, Any]]:
    verification = (
        state.get("verification_result", {})
        .get("output", {})
        .get("verification", {})
    )
    claims = verification.get("claims", []) if isinstance(verification, dict) else []
    return [item for item in claims if isinstance(item, dict)]


def _logistics_plan(state: dict[str, Any]) -> dict[str, Any]:
    payload = state.get("logistics_plan_result", {})
    if isinstance(payload, dict) and isinstance(payload.get("output"), dict):
        payload = payload["output"]
    if isinstance(payload, dict) and isinstance(payload.get("logistics_plan"), dict):
        payload = payload["logistics_plan"]
    return payload if isinstance(payload, dict) else {}


def _failure_context(state: dict[str, Any]) -> dict[str, Any]:
    context = state.get("failure_context")
    if isinstance(context, dict) and context:
        return context
    mode = str(state.get("test_mode", "normal"))
    if mode == "logistics_reroute_failure":
        return {
            "origin": "Controlled TEST ONLY execution-mode injection",
            "business_logic_failure": False,
            "classification": "recoverable",
            "reason": "A transient fulfilment execution timeout was simulated after carrier planning completed.",
            "impact": "The carrier plan remains valid and no simulated fulfilment action was accepted from the failed test agent.",
        }
    if mode == "logistics_stop_failure":
        return {
            "origin": "Controlled TEST ONLY execution-mode injection",
            "business_logic_failure": False,
            "classification": "unrecoverable",
            "reason": "A non-recoverable execution-integrity fault was simulated before fulfilment could continue safely.",
            "impact": "No unverified fulfilment result was allowed to continue to verification or reporting.",
        }
    return {}


def _recovery_context(state: dict[str, Any]) -> dict[str, Any]:
    context = state.get("recovery_context")
    if isinstance(context, dict) and context:
        return context
    recovery = state.get("recovery_result", {}).get("output", {}).get("recovery", {})
    if not isinstance(recovery, dict):
        return {}
    return {
        "action": recovery.get("action"),
        "reason": recovery.get("reason"),
    }


def _add_logistics_appendix(
    story: list[Any],
    state: dict[str, Any],
    styles: dict[str, ParagraphStyle],
) -> None:
    plan = _logistics_plan(state)
    story.extend(_section_title("Logistics Fulfilment Summary", styles))
    if not plan:
        story.append(_p("No logistics plan payload was available in the workflow snapshot.", styles["body"]))
        return

    shipping_plan = plan.get("shipping_plan", [])
    if not isinstance(shipping_plan, list):
        shipping_plan = []
    risk_rows = [
        item
        for item in shipping_plan
        if isinstance(item, dict)
        and str(item.get("delivery_risk", "")).upper() not in {"ON_TIME", "LOW", "NONE", ""}
    ]
    story.append(
        _metadata_table(
            [
                ("Optimization mode", plan.get("optimization_mode", "balanced")),
                ("Orders planned", plan.get("orders_planned", len(shipping_plan))),
                ("Total shipping cost", f"{float(plan.get('total_shipping_cost', 0) or 0):.2f}"),
                ("Average ETA", f"{float(plan.get('average_eta_days', 0) or 0):.1f} days"),
                ("At-risk orders", len(risk_rows)),
            ],
            styles,
        )
    )

    if shipping_plan:
        story.append(_p("Carrier recommendations", styles["h3"]))
        rows: list[list[Any]] = []
        for item in shipping_plan[:12]:
            if not isinstance(item, dict):
                continue
            rows.append(
                [
                    item.get("order_id", ""),
                    item.get("ship_to_region", ""),
                    item.get("chosen_carrier", ""),
                    f"{float(item.get('shipping_cost', 0) or 0):.2f}",
                    f"{int(item.get('eta_days', 0) or 0)} d",
                    item.get("delivery_risk", ""),
                ]
            )
        story.append(
            _table(
                ["Order", "Region", "Carrier", "Cost", "ETA", "Delivery risk"],
                rows,
                [24, 27, 48, 24, 18, 33],
                styles,
            )
        )
        if len(shipping_plan) > 12:
            story.append(
                _p(
                    f"The PDF shows the first 12 of {len(shipping_plan)} planned orders. The JSON report and workflow state retain the complete plan.",
                    styles["small_muted"],
                )
            )

    if risk_rows:
        story.append(_p("At-risk orders", styles["h3"]))
        for item in risk_rows[:10]:
            story.append(
                _bullet(
                    f"{item.get('order_id', 'Order')}: {item.get('delivery_risk', 'risk')} - expected {item.get('expected_delivery_date', 'unknown')}, promised {item.get('promised_date', 'unknown')}.",
                    styles["body"],
                )
            )
    else:
        story.append(_p("No planned order is marked with a delivery-risk status in this run.", styles["body"]))


def _add_recovery_summary(
    story: list[Any],
    state: dict[str, Any],
    styles: dict[str, ParagraphStyle],
) -> None:
    mode = str(state.get("test_mode", "normal") or "normal")
    if mode not in {"logistics_reroute_failure", "logistics_stop_failure"}:
        story.append(
            _callout(
                "Agent / Recovery Summary",
                "Normal logistics execution completed without activating the controlled recovery lane.",
                styles,
                background=PALE_CYAN,
                accent=GREEN,
            )
        )
        return

    failure = _failure_context(state)
    recovery = _recovery_context(state)
    telemetry = state.get("failed_agent_telemetry") or {}
    route = (state.get("recovery_routing") or [{}])[0]
    if not isinstance(route, dict):
        route = {}

    rows = [
        ("Failure origin", failure.get("origin", "Controlled TEST ONLY execution-mode injection")),
        ("Business-data failure", "No" if not failure.get("business_logic_failure", False) else "Yes"),
        ("Failure classification", str(failure.get("classification", "unknown")).title()),
        ("Failure reason", failure.get("reason", state.get("last_error", ""))),
        ("Recovery decision", str(recovery.get("action", state.get("recovery_action", ""))).upper()),
        ("Decision rationale", recovery.get("why") or recovery.get("reason") or failure.get("recovery_policy", "")),
    ]
    if mode == "logistics_reroute_failure":
        rows.extend(
            [
                ("Primary executor", state.get("original_execution_agent_code", "task_execution")),
                ("Failed TEST ONLY agent", telemetry.get("agent_name", state.get("failed_agent_code", ""))),
                ("Replacement agent", recovery.get("replacement_agent_name") or route.get("selected_agent_name", route.get("selected_agent_code", ""))),
                ("Final outcome", recovery.get("final_outcome") or "Replacement execution completed and the workflow continued through verification, governance, quality evaluation and reporting."),
            ]
        )
    else:
        rows.append(
            (
                "Final outcome",
                "CORTEX stopped safely. Verification, TrustGraph, Governance, Human Approval, Quality Evaluation and Executive Report were not executed.",
            )
        )

    story.extend(_section_title("Agent / Recovery Summary", styles))
    story.append(_metadata_table(rows, styles))
    if telemetry:
        story.append(Spacer(1, 6))
        story.append(
            _callout(
                "Failed-agent trust telemetry",
                (
                    f"Trust {float(telemetry.get('trust_before', 0) or 0):.2f} -> "
                    f"{float(telemetry.get('trust_after', 0) or 0):.2f}; Reliability "
                    f"{float(telemetry.get('reliability_before', 0) or 0):.2f} -> "
                    f"{float(telemetry.get('reliability_after', 0) or 0):.2f}. "
                    "The penalty is persisted in the Agent Registry."
                ),
                styles,
                background=PALE_RED,
                accent=RED,
            )
        )


def _add_complaint_assurance(
    story: list[Any],
    state: dict[str, Any],
    styles: dict[str, ParagraphStyle],
) -> None:
    adversarial = next(
        (
            item
            for item in _verification_claims(state)
            if str(item.get("source_agent_code", "")) == "adversarial_test_agent"
        ),
        None,
    )
    story.extend(_section_title("Continuous Adversarial Assurance", styles))
    if not adversarial:
        story.append(
            _p(
                "No adversarial assurance result was available in this workflow snapshot.",
                styles["body"],
            )
        )
        return
    has_invalid_evidence = bool(adversarial.get("invalid_evidence_ids", []))
    status = str(adversarial.get("verification_result", "unknown")).upper()
    story.append(
        _callout(
            "Adversarial claim rejected by verification",
            (
                f"{adversarial.get('finding_id', 'RC-003')} -> {status}. "
                f"Groundedness: {float(adversarial.get('groundedness', 0) or 0):.2f}. "
                f"Hallucination risk: {float(adversarial.get('hallucination_risk', 0) or 0):.2f}. "
                f"Evidence status: {'INVALID' if has_invalid_evidence else 'MISSING'}. "
                "The rejected claim is presented separately from business findings and is excluded from verified findings and executive recommendations."
            ),
            styles,
            background=PALE_AMBER,
            accent=AMBER,
        )
    )


def build_executive_report_pdf(
    *,
    mission: dict[str, Any],
    run: dict[str, Any],
    report: dict[str, Any],
    state: dict[str, Any],
) -> bytes:
    """Create a polished executive PDF for completed complaint or logistics runs."""
    styles = _styles()
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title=_ascii(report.get("title") or "CORTEX Executive Decision Report"),
        author="CORTEX Enterprise AgentOps",
    )
    story: list[Any] = []

    profile = str(state.get("workflow_profile") or "complaint").lower()
    profile_label = "Logistics Intelligence" if profile == "logistics" else "Customer Complaint Intelligence"
    mode_label = _mode_label(state, profile)

    story.append(_p("CORTEX Executive Decision Report", styles["title"]))
    story.append(
        _p(
            report.get("synthetic_data_notice")
            or "Synthetic demonstration output generated by CORTEX.",
            styles["subtitle"],
        )
    )
    story.append(
        _metadata_table(
            [
                ("Mission", mission.get("title", "CORTEX Mission")),
                ("Workflow", profile_label),
                ("Execution mode", mode_label),
                ("Mission status", run.get("status", report.get("mission_status", "unknown"))),
                ("Mission success score", f"{float(report.get('mission_success_score', 0) or 0):.1f}/100"),
                ("Business objective", mission.get("objective", report.get("business_objective", ""))),
            ],
            styles,
        )
    )

    story.extend(_section_title("Executive Summary", styles))
    story.append(_p(report.get("executive_summary", ""), styles["body"]))

    story.extend(_section_title("Verified Findings", styles))
    findings = report.get("key_findings", [])
    finding_rows: list[list[Any]] = []
    if isinstance(findings, list):
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            finding_rows.append(
                [
                    finding.get("category", "Finding"),
                    finding.get("finding", ""),
                    f"{float(finding.get('groundedness', 0) or 0):.2f}",
                    ", ".join(str(value) for value in finding.get("evidence_ids", [])) or "-",
                ]
            )
    if finding_rows:
        story.append(
            _table(
                ["Category", "Verified finding", "Grounded", "Evidence"],
                finding_rows,
                [31, 73, 24, 46],
                styles,
            )
        )
    else:
        story.append(_p("No governance-permitted verified finding is available.", styles["body"]))

    story.extend(_section_title("Prioritised Recommendations", styles))
    recommendations = report.get("recommendations", [])
    rec_rows: list[list[Any]] = []
    if isinstance(recommendations, list):
        for item in recommendations:
            if not isinstance(item, dict):
                continue
            rec_rows.append(
                [
                    item.get("priority", ""),
                    item.get("category", ""),
                    item.get("action", ""),
                    item.get("risk_level", ""),
                    item.get("status", ""),
                ]
            )
    if rec_rows:
        story.append(
            _table(
                ["#", "Category", "Action", "Risk", "Status"],
                rec_rows,
                [11, 31, 70, 22, 40],
                styles,
            )
        )
    else:
        story.append(_p("No approved recommendation is available.", styles["body"]))

    story.extend(_section_title("Governance Decision", styles))
    story.append(
        _callout(
            "Governance status",
            report.get("governance_status", "No governance status was supplied."),
            styles,
            background=PALE_CYAN,
            accent=CYAN,
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        _callout(
            "Next recommended action",
            report.get("next_recommended_action", "No next action was supplied."),
            styles,
            background=PALE,
            accent=NAVY,
        )
    )

    story.extend(_section_title("Risk / Limitations", styles))
    risks = report.get("risks", []) if isinstance(report.get("risks", []), list) else []
    limitations = report.get("limitations", []) if isinstance(report.get("limitations", []), list) else []
    combined = [*(str(value) for value in risks), *(str(value) for value in limitations)]
    if combined:
        for item in combined:
            story.append(_bullet(item, styles["body"]))
    else:
        story.append(_p("No additional risk or limitation was supplied.", styles["body"]))

    if profile == "complaint":
        _add_complaint_assurance(story, state, styles)
    else:
        _add_logistics_appendix(story, state, styles)
        _add_recovery_summary(story, state, styles)

    story.append(Spacer(1, 10))
    story.append(
        _p(
            "This document is an auditable presentation of the completed CORTEX workflow state. Rejected or unapproved claims are not presented as verified business conclusions.",
            styles["small_muted"],
        )
    )

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()


def build_failure_audit_pdf(
    *,
    mission: dict[str, Any],
    run: dict[str, Any],
    state: dict[str, Any],
) -> bytes:
    """Create a failure/recovery audit PDF when a logistics run stops before reporting."""
    styles = _styles()
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title="CORTEX Failure and Recovery Audit",
        author="CORTEX Enterprise AgentOps",
    )
    story: list[Any] = []
    failure = _failure_context(state)
    recovery = _recovery_context(state)
    telemetry = state.get("failed_agent_telemetry") or {}
    skipped = state.get("skipped_stages") or [
        "Claim Verification",
        "TrustGraph",
        "Governance",
        "Human Approval",
        "Quality Evaluation",
        "Executive Report",
    ]

    story.append(_p("CORTEX Failure & Recovery Audit", styles["title"]))
    story.append(
        _p(
            "No executive decision report was generated because CORTEX terminated the logistics workflow before verified decision output could be produced.",
            styles["subtitle"],
        )
    )
    story.append(
        _metadata_table(
            [
                ("Mission", mission.get("title", "CORTEX Mission")),
                ("Workflow", "Logistics Intelligence"),
                ("Execution mode", _mode_label(state, "logistics")),
                ("Mission status", run.get("status", "failed")),
                ("Current stage", run.get("current_stage", state.get("failed_stage", "workflow_stopped"))),
                ("Business objective", mission.get("objective", "")),
            ],
            styles,
        )
    )

    story.extend(_section_title("Controlled Failure Context", styles))
    story.append(
        _callout(
            "Why the agent failed",
            (
                f"Origin: {failure.get('origin', 'Controlled TEST ONLY execution-mode injection')}. "
                f"Reason: {failure.get('reason', state.get('last_error', 'Controlled failure'))} "
                "This is a deliberate recovery demonstration, not a failure caused by the logistics orders, products, carrier data or carrier-selection business logic."
            ),
            styles,
            background=PALE_RED,
            accent=RED,
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        _metadata_table(
            [
                ("Failure classification", str(failure.get("classification", "unrecoverable")).title()),
                ("Failed agent", telemetry.get("agent_name", state.get("failed_agent_code", ""))),
                ("Business impact", failure.get("impact", "No unverified fulfilment output was accepted.")),
                ("Recovery decision", str(recovery.get("action", state.get("recovery_action", "stop"))).upper()),
                ("Why CORTEX stopped", recovery.get("reason", failure.get("recovery_policy", state.get("stopped_reason", "Continuation is not permitted.")))),
            ],
            styles,
        )
    )

    if telemetry:
        story.extend(_section_title("Trust / Reliability Change", styles))
        story.append(
            _table(
                ["Metric", "Before", "After", "Result"],
                [
                    [
                        "Trust",
                        f"{float(telemetry.get('trust_before', 0) or 0):.2f}",
                        f"{float(telemetry.get('trust_after', 0) or 0):.2f}",
                        "Penalty persisted",
                    ],
                    [
                        "Reliability",
                        f"{float(telemetry.get('reliability_before', 0) or 0):.2f}",
                        f"{float(telemetry.get('reliability_after', 0) or 0):.2f}",
                        "Penalty persisted",
                    ],
                ],
                [42, 38, 38, 56],
                styles,
            )
        )

    story.extend(_section_title("Skipped Downstream Stages", styles))
    for item in skipped:
        label = _ascii(item).replace("_", " ").title()
        story.append(_bullet(f"{label} - SKIPPED / NOT EXECUTED", styles["body"]))

    timeline = run.get("timeline", [])
    if isinstance(timeline, list) and timeline:
        story.extend(_section_title("Execution Audit Trail", styles))
        timeline_rows: list[list[Any]] = []
        for event in timeline[-14:]:
            if not isinstance(event, dict):
                continue
            timeline_rows.append(
                [
                    str(event.get("stage", "workflow")).replace("_", " ").title(),
                    str(event.get("status", "unknown")).upper(),
                    event.get("message", ""),
                ]
            )
        if timeline_rows:
            story.append(
                _table(
                    ["Stage", "Status", "Audit message"],
                    timeline_rows,
                    [42, 28, 104],
                    styles,
                )
            )

    story.append(Spacer(1, 10))
    story.append(
        _callout(
            "Safe termination",
            "CORTEX intentionally produced no executive recommendation because the workflow did not reach verification, governance and quality gates. This audit PDF records the failure and recovery decision without representing incomplete output as a business conclusion.",
            styles,
            background=PALE_AMBER,
            accent=AMBER,
        )
    )

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()
