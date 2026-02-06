#!/usr/bin/env python3
"""
Omvyx Voice — Call Simulator

Runs a multi-turn conversation against the LangGraph workflow directly
(no HTTP server needed).  Demonstrates:

    1. Greeting
    2. Slot-filling (name → DNI → email → phone)
    3. FAQ interruption mid-slot-filling (and resumption)
    4. Appointment booking with unavailable → alternative flow
    5. Goodbye

Usage:
    python simulate_call.py              # run the default scripted scenario
    python simulate_call.py --interactive  # interactive REPL mode
"""

from __future__ import annotations

import asyncio
import sys

from langchain_core.messages import AIMessage, HumanMessage

from graph.workflow import SYSTEM_PROMPT, compile_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CALL_ID = "sim-call-001"

def _last_ai_message(result: dict) -> str:
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage):
            return msg.content
    return "(no response)"


async def send(graph, text: str, *, config: dict) -> str:
    """Send a user utterance and return the agent reply."""
    result = await graph.ainvoke(
        {
            "messages": [SYSTEM_PROMPT, HumanMessage(content=text)],
            "call_id": CALL_ID,
        },
        config=config,
    )
    return _last_ai_message(result)


def banner(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Scripted scenario
# ---------------------------------------------------------------------------

SCENARIO: list[tuple[str, str]] = [
    # (label, user utterance)
    ("1. Greeting", "Hola, buenos días"),
    ("2. Provide name", "Me llamo Ana Torres"),
    ("3. FAQ interruption (location)", "¿Dónde están ubicados?"),
    ("4. Provide DNI", "Mi DNI es 99887766C"),
    ("5. Provide email", "ana.torres@gmail.com"),
    ("6. Provide phone", "+34 611 222 333"),
    ("7. Request appointment (busy slot)", "Quiero una cita para 2026-02-09 10:00"),
    ("8. Pick alternative slot", "Mejor el 2026-02-09 12:00"),
    ("9. Goodbye", "Eso es todo, adiós"),
]


async def run_scripted():
    banner("OMVYX VOICE — Scripted Call Simulation")
    graph = compile_graph()
    config = {"configurable": {"thread_id": CALL_ID}}

    for label, utterance in SCENARIO:
        print(f"\n--- {label} ---")
        print(f"  USER : {utterance}")
        reply = await send(graph, utterance, config=config)
        print(f"  AGENT: {reply}")

    # Print final state
    state = await graph.aget_state(config)
    print("\n" + "="*60)
    print("  FINAL STATE SNAPSHOT")
    print("="*60)
    vals = state.values
    print(f"  user_found    : {vals.get('user_found')}")
    print(f"  user_profile  : {vals.get('user_profile')}")
    print(f"  missing_fields: {vals.get('missing_fields')}")
    print(f"  booking       : {vals.get('booking')}")
    print()


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

async def run_interactive():
    banner("OMVYX VOICE — Interactive Mode (type 'quit' to exit)")
    graph = compile_graph()
    config = {"configurable": {"thread_id": CALL_ID}}

    while True:
        try:
            text = input("\n  YOU: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!")
            break
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue

        reply = await send(graph, text, config=config)
        print(f"  OMVYX: {reply}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--interactive" in sys.argv:
        asyncio.run(run_interactive())
    else:
        asyncio.run(run_scripted())
