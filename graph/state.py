"""
Omvyx Voice — Graph State Definition

This is the single source of truth for all conversational state.
LangGraph persists this via its Checkpointer between stateless HTTP requests,
keyed by Retell's call_id (used as the LangGraph thread_id).
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from langgraph.graph import MessagesState


# ---------------------------------------------------------------------------
# Sub-models used inside the main state
# ---------------------------------------------------------------------------

class UserProfile(BaseModel):
    """Collected user data.  None means "not yet captured"."""
    name: str | None = None
    dni: str | None = None
    email: str | None = None
    phone: str | None = None


class BookingRequest(BaseModel):
    """Temporary holder for an in-progress appointment booking."""
    requested_date: str | None = None
    confirmed_slot: str | None = None
    status: Literal["idle", "checking", "offered", "confirmed", "failed"] = "idle"


# ---------------------------------------------------------------------------
# Main graph state
# ---------------------------------------------------------------------------

class OmvyxState(MessagesState):
    """
    Extends LangGraph's built-in MessagesState (which gives us `messages`
    with the standard append-reducer) and adds domain-specific fields.

    Key design decisions:
    - user_found: drives the routing — True bypasses slot-filling entirely.
    - missing_fields: the slot-filling loop checks this list; when empty,
      all required data has been captured and we can proceed.
    - interrupted_by_faq: set when the user asks a FAQ *during* data
      collection so the graph can answer it and immediately resume
      collecting the next missing field.
    """

    # Call metadata
    call_id: str = ""

    # User identification
    user_found: bool = False
    user_profile: UserProfile = Field(default_factory=UserProfile)

    # Slot-filling control
    missing_fields: list[str] = Field(
        default_factory=lambda: ["name", "dni", "email", "phone"]
    )
    current_slot: str = ""
    interrupted_by_faq: bool = False

    # Booking
    booking: BookingRequest = Field(default_factory=BookingRequest)

    # Routing hint set by the router node
    intent: Literal[
        "greeting", "faq", "collect_data", "booking", "end_call", "unknown"
    ] = "unknown"
