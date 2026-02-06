"""
Omvyx Voice — LangGraph Workflow

This is the brain of the AI Receptionist.  It defines the conversational
graph that processes every Retell webhook invocation.

Architecture overview:
    ┌──────────┐
    │  router  │  ← entry point: classifies user intent
    └────┬─────┘
         │ conditional edges
    ┌────┴────┬──────────┬──────────────┐
    ▼         ▼          ▼              ▼
 greet    handle_faq  collect_data  manage_booking
    │         │          │              │
    │         │          ▼              │
    │         │   ┌─────────────┐      │
    │         │   │ still       │      │
    │         │   │ missing? ───┼─► loops back to collect_data
    │         │   └──────┬──────┘
    │         │          │ all captured
    │         ▼          ▼              ▼
    └─────────┴──► respond ◄────────────┘

CRITICAL — Persistence:
    The graph is compiled with a MemorySaver checkpointer.  Every
    invocation receives `config={"configurable": {"thread_id": call_id}}`
    so state is resumed from the exact point where the previous webhook
    request left off.  This is how we survive the stateless HTTP boundary.

NOTE:
    LangGraph passes state as a plain dict to node functions (even when
    the schema is a Pydantic model).  All nodes therefore use dict-style
    access: state["messages"], state["user_profile"], etc.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from graph.state import BookingRequest, OmvyxState, UserProfile
from tools.calendar import book_slot, check_availability
from tools.crm import create_user, lookup_user
from tools.faq import search_faq


# ---------------------------------------------------------------------------
# System prompt — personality of the receptionist
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = SystemMessage(content="""\
Eres Omvyx, una recepcionista virtual profesional y amable.
Hablas en español de España de forma natural y cercana.

REGLAS:
- Sé concisa: las respuestas van por voz, así que máximo 2-3 frases.
- Nunca inventes datos. Si no sabes algo, dilo.
- Si estás recogiendo datos del usuario y te interrumpen con una pregunta,
  responde la pregunta y VUELVE INMEDIATAMENTE a pedir el dato que faltaba.
- Cuando ofrezcas citas alternativas, di las opciones de forma clara.
""")


# ---------------------------------------------------------------------------
# FIELD_PROMPTS — what to say when asking for each slot
# ---------------------------------------------------------------------------

FIELD_PROMPTS: dict[str, str] = {
    "name": "Para poder ayudarle, necesito su nombre completo, ¿me lo puede indicar?",
    "dni": "Perfecto. ¿Me puede facilitar su DNI o documento de identidad?",
    "email": "Genial. ¿Cuál es su dirección de correo electrónico?",
    "phone": "Y por último, ¿un número de teléfono de contacto?",
}


# ---------------------------------------------------------------------------
# State access helpers
# ---------------------------------------------------------------------------

def _get_profile(state: dict) -> UserProfile:
    """Reconstruct UserProfile from the state dict."""
    raw = state.get("user_profile")
    if isinstance(raw, UserProfile):
        return raw
    if isinstance(raw, dict):
        return UserProfile(**raw)
    return UserProfile()


def _get_booking(state: dict) -> BookingRequest:
    """Reconstruct BookingRequest from the state dict."""
    raw = state.get("booking")
    if isinstance(raw, BookingRequest):
        return raw
    if isinstance(raw, dict):
        return BookingRequest(**raw)
    return BookingRequest()


def _last_human_text(state: dict) -> str:
    """Extract the text of the most recent HumanMessage."""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return m.content
    return ""


# ---------------------------------------------------------------------------
# Extraction helpers — pull structured data from free-text user utterances
# ---------------------------------------------------------------------------

def _extract_field_value(field: str, text: str) -> str | None:
    """
    Best-effort extraction of a slot value from the user's last message.
    In production you'd use an LLM function-call or NER model here.
    """
    text = text.strip()
    if not text:
        return None

    if field == "dni":
        m = re.search(r"\b(\d{7,8}\s*[A-Za-z])\b", text)
        return m.group(1).replace(" ", "").upper() if m else None

    if field == "email":
        m = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
        return m.group(0).lower() if m else None

    if field == "phone":
        digits = re.sub(r"[^\d+]", "", text)
        if len(digits) >= 9:
            return digits
        return None

    if field == "name":
        # Try explicit patterns first: "me llamo X", "soy X", "mi nombre es X"
        name_patterns = [
            r"(?:me llamo|soy|mi nombre es)\s+(.+)",
        ]
        for pat in name_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                name_part = m.group(1).strip().rstrip(".")
                words = [w for w in name_part.split() if w.isalpha()]
                if words:
                    return " ".join(words).title()

        # Fallback: accept if the entire message looks like a name (≥2 words)
        words = [w for w in text.split() if w.isalpha()]
        if len(words) >= 2 and len(words) <= 4:
            return " ".join(words).title()
        if len(words) == 1 and len(words[0]) > 1:
            return words[0].title()
        return None

    return None


def _detect_intent(text: str) -> str:
    """Lightweight keyword-based intent classifier."""
    t = text.lower()

    greetings = ["hola", "buenos días", "buenas tardes", "buenas noches", "hey", "hello"]
    if any(g in t for g in greetings) and len(t.split()) <= 4:
        return "greeting"

    faq_signals = [
        "dónde", "donde", "ubicación", "dirección", "horario", "hora",
        "precio", "costo", "cancelar", "seguro", "parking", "location",
        "where", "address", "hours", "price", "cancel", "insurance",
    ]
    if any(s in t for s in faq_signals):
        return "faq"

    booking_signals = ["cita", "reservar", "agendar", "appointment", "book", "turno"]
    if any(s in t for s in booking_signals):
        return "booking"

    bye_signals = ["adiós", "chao", "bye", "hasta luego", "nada más", "eso es todo"]
    if any(s in t for s in bye_signals):
        return "end_call"

    return "collect_data"


# ===================================================================
# GRAPH NODES
#
# Every node receives `state` as a plain dict and returns a partial
# dict of updates to merge back into the state.
# ===================================================================

async def router(state: dict) -> dict:
    """
    Entry node — classifies the latest user message and sets `intent`.
    Slot extraction only runs when the user is actively providing data
    (intent == "collect_data") to avoid false positives on greetings.
    """
    last_msg = _last_human_text(state)
    intent = _detect_intent(last_msg)

    updates: dict[str, Any] = {}
    profile = _get_profile(state).model_copy()
    remaining = list(state.get("missing_fields", ["name", "dni", "email", "phone"]))
    booking = _get_booking(state)

    # --- Override intent when we're waiting for a booking response ---
    # If the agent offered alternative slots and the user replies,
    # treat it as a booking continuation even without booking keywords.
    if booking.status == "offered" and intent == "collect_data":
        intent = "booking"

    # --- Slot extraction only when user is answering data questions ---
    # Avoids false positives (e.g. "Hola, buenos días" → name).
    if intent == "collect_data" or intent == "faq":
        for field in list(remaining):
            val = _extract_field_value(field, last_msg)
            if val:
                setattr(profile, field, val)
                remaining.remove(field)

    updates["user_profile"] = profile
    updates["missing_fields"] = remaining
    updates["intent"] = intent

    # If we were in the middle of slot-filling and user asked a FAQ, flag it
    if intent == "faq" and state.get("current_slot"):
        updates["interrupted_by_faq"] = True

    # If user provided a DNI, try CRM lookup
    if profile.dni and not state.get("user_found", False):
        crm_record = await lookup_user(profile.dni)
        if crm_record:
            updates["user_found"] = True
            updates["user_profile"] = UserProfile(**crm_record)
            updates["missing_fields"] = []

    return updates


async def greet(state: dict) -> dict:
    """Generate a greeting response."""
    profile = _get_profile(state)
    name = profile.name or ""
    if name:
        text = f"¡Hola, {name}! Bienvenido/a a Omvyx. ¿En qué puedo ayudarle hoy?"
    else:
        text = "¡Hola! Bienvenido/a a Omvyx. ¿En qué puedo ayudarle hoy?"
    return {"messages": [AIMessage(content=text)]}


async def handle_faq(state: dict) -> dict:
    """
    Answer a FAQ question.  If the user was in the middle of slot-filling
    (interrupted_by_faq=True), append a prompt to resume data collection.
    """
    last_msg = _last_human_text(state)

    answer = await search_faq(last_msg)
    if not answer:
        answer = "Lo siento, no tengo información sobre eso. ¿Puedo ayudarle con algo más?"

    # Resume slot-filling after answering the FAQ
    missing = state.get("missing_fields", [])
    if state.get("interrupted_by_faq") and missing:
        next_field = missing[0]
        answer += f" Pero volviendo a sus datos, {FIELD_PROMPTS[next_field].lower()}"

    return {
        "messages": [AIMessage(content=answer)],
        "interrupted_by_faq": False,
    }


async def collect_data(state: dict) -> dict:
    """
    Slot-filling node.  Checks `missing_fields` and asks for the next one.
    If the current_slot was just answered (extracted in router), moves on.
    When all fields are captured, registers the user in the CRM.
    """
    missing = state.get("missing_fields", [])
    profile = _get_profile(state)

    # All captured — register user
    if not missing:
        if not state.get("user_found", False):
            await create_user(
                name=profile.name or "",
                dni=profile.dni or "",
                email=profile.email or "",
                phone=profile.phone or "",
            )
        text = (
            f"Perfecto, {profile.name}. Ya tengo todos sus datos registrados. "
            "¿Le gustaría reservar una cita?"
        )
        return {
            "messages": [AIMessage(content=text)],
            "user_found": True,
            "current_slot": "",
        }

    # Ask for the next missing field
    next_field = missing[0]
    prompt_text = FIELD_PROMPTS[next_field]

    return {
        "messages": [AIMessage(content=prompt_text)],
        "current_slot": next_field,
    }


async def manage_booking(state: dict) -> dict:
    """
    Handle appointment booking.  If user data isn't complete yet,
    redirect to data collection first.
    """
    missing = state.get("missing_fields", [])
    profile = _get_profile(state)

    # Need user data before booking
    if missing:
        text = (
            "Con gusto le ayudo a reservar una cita. "
            f"Pero primero necesito algunos datos. {FIELD_PROMPTS[missing[0]]}"
        )
        return {
            "messages": [AIMessage(content=text)],
            "intent": "collect_data",
            "current_slot": missing[0],
        }

    booking = _get_booking(state)

    # Extract date from last message if not yet set
    if not booking.requested_date:
        last_msg = _last_human_text(state)
        date_match = re.search(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}", last_msg)
        if date_match:
            booking = BookingRequest(
                requested_date=date_match.group(0), status="checking"
            )
        else:
            text = (
                "¿Para qué fecha y hora le gustaría la cita? "
                "Por ejemplo: 2026-02-11 10:00"
            )
            return {"messages": [AIMessage(content=text)]}

    # Check availability
    if booking.status == "checking":
        result = await check_availability(booking.requested_date)
        if result.get("available"):
            await book_slot(booking.requested_date, profile.dni or "")
            booking = BookingRequest(
                requested_date=booking.requested_date,
                confirmed_slot=booking.requested_date,
                status="confirmed",
            )
            text = (
                f"¡Listo! Su cita ha quedado confirmada para el "
                f"{booking.confirmed_slot}. ¿Necesita algo más?"
            )
        else:
            alts = result.get("alternatives", [])
            booking = BookingRequest(
                requested_date=booking.requested_date, status="offered"
            )
            if alts:
                options = " o ".join(alts)
                text = (
                    f"Lo siento, el {booking.requested_date} no está disponible. "
                    f"Tengo disponibilidad el {options}. "
                    "¿Le viene bien alguna de estas opciones?"
                )
            else:
                error = result.get("error", "No hay disponibilidad.")
                text = f"Lo siento, {error} ¿Desea intentar otra fecha?"

        return {
            "messages": [AIMessage(content=text)],
            "booking": booking,
        }

    # If we offered alternatives and user responds
    if booking.status == "offered":
        last_msg = _last_human_text(state)
        date_match = re.search(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}", last_msg)
        if date_match:
            slot = date_match.group(0)
            result = await check_availability(slot)
            if result.get("available"):
                await book_slot(slot, profile.dni or "")
                booking = BookingRequest(
                    requested_date=slot,
                    confirmed_slot=slot,
                    status="confirmed",
                )
                text = (
                    f"¡Perfecto! Cita confirmada para el {slot}. "
                    "¿Algo más en lo que pueda ayudarle?"
                )
            else:
                text = "Ese horario tampoco está disponible. ¿Quiere probar otra fecha?"
        else:
            booking = BookingRequest(status="idle")
            text = "Entendido. ¿Para qué fecha y hora le gustaría la cita?"

        return {
            "messages": [AIMessage(content=text)],
            "booking": booking,
        }

    # Fallback
    return {
        "messages": [AIMessage(content="¿Le gustaría reservar una cita? Dígame la fecha y hora deseada.")],
        "booking": BookingRequest(status="idle"),
    }


async def end_call(state: dict) -> dict:
    """Polite goodbye."""
    profile = _get_profile(state)
    name = profile.name or ""
    if name:
        text = f"Ha sido un placer atenderle, {name}. ¡Hasta pronto!"
    else:
        text = "Gracias por llamar a Omvyx. ¡Hasta pronto!"
    return {"messages": [AIMessage(content=text)]}


async def respond(state: dict) -> dict:
    """
    Terminal passthrough — exists so all branches have a clean
    convergence point before END.  Does nothing; the actual response
    message was already appended by the preceding node.
    """
    return {}


# ===================================================================
# ROUTING LOGIC
#
# Routing functions also receive state as a plain dict.
# ===================================================================

def route_after_router(state: dict) -> str:
    """Conditional edge from `router` → next node."""
    intent = state.get("intent", "unknown")
    missing = state.get("missing_fields", [])

    # If slot-filling is active and user didn't ask a FAQ or booking,
    # stay in collect_data regardless of what was detected
    if missing and intent not in ("faq", "booking", "end_call", "greeting"):
        return "collect_data"
    return intent


def route_after_collect(state: dict) -> str:
    """After collect_data, go to respond (wait for next user input)."""
    return "respond"


def route_after_booking(state: dict) -> str:
    """After manage_booking, redirect to collect_data if intent was changed."""
    if state.get("intent") == "collect_data":
        return "collect_data"
    return "respond"


# ===================================================================
# GRAPH ASSEMBLY
# ===================================================================

def build_graph() -> StateGraph:
    """
    Construct the LangGraph StateGraph (uncompiled).
    """
    builder = StateGraph(OmvyxState)

    # --- Nodes ---
    builder.add_node("router", router)
    builder.add_node("greet", greet)
    builder.add_node("handle_faq", handle_faq)
    builder.add_node("collect_data", collect_data)
    builder.add_node("manage_booking", manage_booking)
    builder.add_node("end_call", end_call)
    builder.add_node("respond", respond)

    # --- Entry ---
    builder.set_entry_point("router")

    # --- Conditional edges from router ---
    builder.add_conditional_edges(
        "router",
        route_after_router,
        {
            "greeting": "greet",
            "faq": "handle_faq",
            "collect_data": "collect_data",
            "booking": "manage_booking",
            "end_call": "end_call",
            "unknown": "collect_data",
        },
    )

    # --- Edges from action nodes → respond → END ---
    builder.add_edge("greet", "respond")
    builder.add_edge("handle_faq", "respond")
    builder.add_edge("end_call", "respond")

    builder.add_conditional_edges(
        "collect_data",
        route_after_collect,
        {"respond": "respond"},
    )

    builder.add_conditional_edges(
        "manage_booking",
        route_after_booking,
        {
            "respond": "respond",
            "collect_data": "collect_data",
        },
    )

    builder.add_edge("respond", END)

    return builder


def compile_graph(checkpointer=None):
    """
    Build and compile the graph with checkpointing.

    Args:
        checkpointer: A LangGraph checkpointer instance.  Defaults to
                      MemorySaver (in-memory, fine for dev/single-process).
                      For production, pass AsyncSqliteSaver or a Redis-backed
                      checkpointer.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    builder = build_graph()
    return builder.compile(checkpointer=checkpointer)
