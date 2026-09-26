from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.services.llm import get_llm_service


CortexIntent = Literal[
    "conversation",
    "cortex_help",
    "complaint",
    "logistics",
    "unsupported",
]


LOGISTICS_STRONG_TERMS = (
    "logistics",
    "shipment",
    "shipping",
    "carrier",
    "fulfilment",
    "fulfillment",
    "warehouse",
    "dispatch",
    "supply chain",
    "freight",
    "last mile",
    "last-mile",
    "routing plan",
    "route optimisation",
    "route optimization",
    "pending order",
    "pending orders",
    "orders at risk",
    "order at risk",
)

LOGISTICS_SUPPORTING_TERMS = (
    "order",
    "orders",
    "delivery",
    "eta",
    "weight",
    "sku",
    "shipping cost",
    "promised date",
    "products",
    "deliveries",
    "delayed delivery",
    "delayed deliveries",
)

COMPLAINT_TERMS = (
    "complaint",
    "complaints",
    "customer complaint",
    "customer complaints",
    "customer grievance",
    "grievance",
    "refund issue",
    "refund issues",
    "support ticket",
    "support tickets",
    "customer dissatisfaction",
    "complaint category",
    "complaint categories",
    "customer feedback",
    "customer issues",
    "customer issue",
)

GREETING_EXACT = {
    "hi",
    "hii",
    "hiii",
    "hiiii",
    "hlo",
    "hello",
    "hey",
    "hey there",
    "hello there",
    "hi cortex",
    "hello cortex",
    "hey cortex",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "how r u",
    "how are u",
    "how do you do",
    "whats up",
    "what s up",
    "what's up",
    "how r you",
}

CORTEX_HELP_PHRASES = (
    "who are you",
    "what are you",
    "tell me who you are",
    "tell me about yourself",
    "tell me about cortex",
    "can you tell me who you are",
    "can u tell me who you are",
    "who r u",
    "what is cortex",
    "what does cortex do",
    "what can you do",
    "what can u do",
    "what can cortex do",
    "what do you support",
    "what does cortex support",
    "your capabilities",
    "cortex capabilities",
    "how does cortex work",
    "how do i use cortex",
    "help me use cortex",
)


class CortexIntentLLMOutput(BaseModel):
    """Small structured response used only by the Create Mission intent gateway."""

    intent: CortexIntent
    response: str = Field(min_length=1, max_length=500)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


def _value(source: Any, key: str) -> str:
    if isinstance(source, dict):
        return str(source.get(key, "") or "")
    return str(getattr(source, key, "") or "")


def _normalized_objective(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _conversation_response(normalized: str) -> str:
    if "how" in normalized and ("are you" in normalized or "r u" in normalized):
        return "I'm ready to help. You can ask about CORTEX or describe a Customer Complaint or Logistics business objective."
    return "Hi! I'm CORTEX. I can help with Customer Complaint Intelligence, Logistics Intelligence, or questions about the CORTEX platform."


def _help_response() -> str:
    return (
        "I'm CORTEX, an Enterprise AgentOps Mission Control Platform. "
        "I coordinate specialized agents for planning, routing, verification, governance, recovery and reporting across Customer Complaint and Logistics Intelligence workflows."
    )


def _unsupported_response() -> str:
    return (
        "That request is outside CORTEX's supported enterprise scope. "
        "I can assist with CORTEX platform questions, Customer Complaint Intelligence, or Logistics Intelligence."
    )


def _deterministic_intent(objective: str) -> CortexIntent | None:
    """Resolve only high-confidence objective intent before using the LLM.

    The dropdown is not used to *classify* the text. That keeps greetings and
    unrelated questions from being forced into a workflow. Mission creation later
    compares the detected business intent with the user's selected Business Domain.
    """

    normalized = _normalized_objective(objective)
    if not normalized:
        return "unsupported"

    if normalized in GREETING_EXACT:
        return "conversation"

    if any(phrase in normalized for phrase in CORTEX_HELP_PHRASES):
        return "cortex_help"

    logistics_strong_hits = sum(term in normalized for term in LOGISTICS_STRONG_TERMS)
    logistics_supporting_hits = sum(term in normalized for term in LOGISTICS_SUPPORTING_TERMS)
    complaint_hits = sum(term in normalized for term in COMPLAINT_TERMS)

    # Explicit complaint language stays in the complaint lane even when the text
    # mentions a delivery/order issue as the subject of the complaint.
    if complaint_hits >= 1:
        return "complaint"

    if logistics_strong_hits >= 1 and logistics_supporting_hits >= 1:
        return "logistics"
    if logistics_strong_hits >= 2 or logistics_supporting_hits >= 3:
        return "logistics"

    # A short greeting prefix is conversational only when it is not followed by a
    # recognizable business request (for example, "hi analyse pending orders").
    if normalized.startswith(("hi ", "hii ", "hlo ", "hello ", "hey ")) and len(normalized.split()) <= 6:
        return "conversation"

    return None


def _fallback_output(objective: str) -> CortexIntentLLMOutput:
    normalized = _normalized_objective(objective)
    intent = _deterministic_intent(objective) or "unsupported"

    if intent == "conversation":
        response = _conversation_response(normalized)
    elif intent == "cortex_help":
        response = _help_response()
    elif intent == "complaint":
        response = "Customer Complaint Intelligence detected. CORTEX can create the mission and continue with the complaint data-validation gate."
    elif intent == "logistics":
        response = "Logistics Intelligence detected. CORTEX can create the mission and continue with the logistics data-validation gate."
    else:
        response = _unsupported_response()

    return CortexIntentLLMOutput(
        intent=intent,
        response=response,
        confidence=1.0 if intent != "unsupported" else 0.65,
    )


def classify_cortex_input(objective: str) -> dict[str, Any]:
    """Classify Create Mission input before a mission is persisted.

    Returns one of five intents:
    conversation, cortex_help, complaint, logistics or unsupported.

    Greetings and CORTEX identity/help questions are handled locally for guaranteed,
    fast behavior. Ambiguous natural-language inputs are classified by the shared
    CORTEX LLM service. Unsupported questions receive a controlled scope response;
    their requested content is never answered and no workflow is launched.
    """

    objective_text = str(objective or "").strip()
    normalized = _normalized_objective(objective_text)

    if not objective_text:
        return {
            "intent": "unsupported",
            "message": "Enter a business objective or ask a question about CORTEX.",
            "confidence": 1.0,
            "provider": "local",
            "tokens": 0,
            "latency_ms": 0,
        }

    deterministic = _deterministic_intent(objective_text)
    if deterministic in {"conversation", "cortex_help", "complaint", "logistics"}:
        fallback = _fallback_output(objective_text)
        return {
            "intent": fallback.intent,
            "message": fallback.response,
            "confidence": fallback.confidence,
            "provider": "local",
            "tokens": 0,
            "latency_ms": 0,
        }

    fallback = _fallback_output(objective_text)
    llm = get_llm_service()
    result = llm.invoke_structured(
        agent_name="cortex_intent_gateway",
        schema=CortexIntentLLMOutput,
        system_prompt=(
            "You are the CORTEX intent gateway. Classify ONLY the user's Business objective. "
            "Ignore any preselected dropdown, title, priority, risk or budget. "
            "Choose exactly one intent: conversation, cortex_help, complaint, logistics, unsupported. "
            "conversation = greeting or lightweight small talk such as hi/hello/how are you. "
            "cortex_help = questions about CORTEX identity, capabilities, supported workflows or how it works. "
            "complaint = a real enterprise objective about customer complaints, grievances, complaint patterns, refund/support issues or complaint root causes. "
            "logistics = a real enterprise objective about orders, shipments, carriers, fulfilment, delivery risk, routing, cost, ETA, warehouse or supply-chain decisions. "
            "unsupported = every other topic, including sports, news, jokes, weather, general knowledge, entertainment, politics, coding or unrelated questions. "
            "Do not answer unsupported questions. For unsupported, only state that the request is outside CORTEX's supported enterprise scope. "
            "For conversation/help, respond naturally in at most two short sentences. "
            "For complaint/logistics, only confirm the detected CORTEX workflow."
        ),
        user_prompt=objective_text,
        mock_value=fallback,
    )

    if result.status != "completed" or not isinstance(result.content, CortexIntentLLMOutput):
        parsed = fallback
        provider = result.provider
        tokens = int(result.usage.total_tokens)
        latency_ms = int(result.latency_ms)
    else:
        parsed = result.content
        provider = result.provider
        tokens = int(result.usage.total_tokens)
        latency_ms = int(result.latency_ms)

    # Never let an unsupported question receive the factual answer it asked for.
    if parsed.intent == "unsupported":
        message = _unsupported_response()
    elif parsed.intent == "conversation":
        message = parsed.response.strip() or _conversation_response(normalized)
    elif parsed.intent == "cortex_help":
        message = parsed.response.strip() or _help_response()
    elif parsed.intent == "complaint":
        message = "Customer Complaint Intelligence detected. CORTEX can create the mission and continue with the complaint data-validation gate."
    else:
        message = "Logistics Intelligence detected. CORTEX can create the mission and continue with the logistics data-validation gate."

    return {
        "intent": parsed.intent,
        "message": message,
        "confidence": float(parsed.confidence),
        "provider": provider,
        "tokens": tokens,
        "latency_ms": latency_ms,
    }


def detect_workflow_profile(mission: Any) -> str:
    """Return the workflow lane for an already-created supported mission.

    For persisted missions the selected Business Domain is authoritative. The
    objective classifier is used *before* persistence to validate compatibility, so
    the runtime must never silently switch a Complaint mission into Logistics (or
    vice versa) after the user has made an explicit domain choice.
    """

    domain = _value(mission, "business_domain").strip().lower()
    if "logistics" in domain:
        return "logistics"
    if "complaint" in domain:
        return "complaint"

    # Compatibility fallback for legacy records without one of the two supported
    # domain labels. New Create Mission submissions never rely on this path.
    objective = _value(mission, "objective")
    detected = _deterministic_intent(objective)
    return "logistics" if detected == "logistics" else "complaint"
