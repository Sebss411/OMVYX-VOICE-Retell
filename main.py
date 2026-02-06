"""
Omvyx Voice — FastAPI Server

Exposes a single POST /retell-webhook endpoint that receives Retell AI
payloads and routes them through the LangGraph workflow.

Key design:
    - The Retell `call_id` is used as LangGraph's `thread_id` so the
      checkpointer can restore the full conversation state on every
      stateless HTTP request.
    - The graph is compiled once at startup with a MemorySaver checkpointer.
    - For production, swap MemorySaver with AsyncSqliteSaver (or Redis)
      so state survives server restarts.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage

from graph.workflow import SYSTEM_PROMPT, compile_graph

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("omvyx")

# ---------------------------------------------------------------------------
# App + Graph
# ---------------------------------------------------------------------------

app = FastAPI(title="Omvyx Voice", version="1.0.0")
graph = compile_graph()  # single process — MemorySaver is fine


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Retell Webhook
# ---------------------------------------------------------------------------

@app.post("/retell-webhook")
async def retell_webhook(request: Request):
    """
    Retell sends a JSON payload on every conversational turn.

    Expected payload shape (simplified):
        {
            "call_id": "...",
            "event": "call_started" | "call_ended" | "call_analyzed",
            "transcript": [
                {"role": "agent", "content": "..."},
                {"role": "user",  "content": "..."}
            ]
        }

    We extract the last user utterance, invoke the LangGraph workflow
    with the call_id as thread_id, and return the agent's response.
    """
    body = await request.json()
    call_id: str = body.get("call_id", "unknown")
    event: str = body.get("event", "")

    logger.info("Webhook received — call_id=%s  event=%s", call_id, event)

    # Retell fires events for lifecycle stages we don't need to respond to
    if event in ("call_ended", "call_analyzed"):
        return JSONResponse({"response_id": call_id, "content": ""})

    # Extract the last user utterance from the transcript
    transcript: list[dict] = body.get("transcript", [])
    user_text = ""
    for turn in reversed(transcript):
        if turn.get("role") == "user":
            user_text = turn.get("content", "")
            break

    if not user_text:
        # First turn (call_started with no transcript yet) — greet
        user_text = "Hola"

    # Invoke graph with checkpointing via thread_id = call_id
    config = {"configurable": {"thread_id": call_id}}
    input_state = {
        "messages": [SYSTEM_PROMPT, HumanMessage(content=user_text)],
        "call_id": call_id,
    }

    result = await graph.ainvoke(input_state, config=config)

    # The last AI message is the response to send back to Retell
    agent_reply = ""
    for msg in reversed(result.get("messages", [])):
        if hasattr(msg, "content") and msg.type == "ai":
            agent_reply = msg.content
            break

    logger.info("Responding — call_id=%s  reply=%s", call_id, agent_reply[:80])

    # Retell expects this response shape
    return JSONResponse({
        "response_id": call_id,
        "content": agent_reply,
        "content_complete": True,
        "end_call": False,
    })


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
