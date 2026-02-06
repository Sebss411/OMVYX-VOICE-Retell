"""
Omvyx Voice — FastAPI Server (WebSocket)

Exposes a WebSocket /retell-webhook endpoint that handles real-time
bidirectional communication with Retell AI.

Architecture:
    - NON-BLOCKING: Graph inference runs in asyncio.create_task(), never
      blocking the main receive loop.  This allows the loop to process
      interrupt signals immediately.
    - The Retell call_id is used as LangGraph's thread_id so the
      checkpointer restores full conversation state across turns.
    - graph/workflow.py is treated as IMMUTABLE (read-only).
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain_core.messages import HumanMessage, SystemMessage

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
# Retell WebSocket Endpoint
# ---------------------------------------------------------------------------


@app.websocket("/retell-webhook/{call_id}")
async def retell_websocket(ws: WebSocket, call_id: str):
    """
    Real-time bidirectional communication with Retell AI.

    Retell connects to /retell-webhook/{call_id} where call_id identifies
    the active call.  This ID is used as LangGraph's thread_id for
    checkpointer persistence.

    NON-BLOCKING ARCHITECTURE:
        The main while-loop only reads incoming WebSocket frames and
        dispatches work.  All graph inference runs inside asyncio.Task
        objects so the loop is always free to process interrupts.

    Retell events handled:
        - interaction_begin / call_details → optional agent greeting
        - response_required               → cancel current + launch new task
        - interrupt / update_only          → cancel current task immediately
    """
    await ws.accept()
    logger.info("WebSocket connected — call_id=%s", call_id)

    current_task: asyncio.Task | None = None

    # ---------------------------------------------------------------
    # Generation handler — runs inside its own asyncio.Task
    # ---------------------------------------------------------------

    async def handle_generation(
        transcript: list[dict],
        response_id: int,
        cid: str,
    ) -> None:
        """
        Run the LangGraph workflow and send the response to Retell.

        This coroutine is always wrapped in asyncio.create_task() so it
        can be cancelled when the user interrupts.

        The graph is deterministic (no LLM token streaming), so we use
        ainvoke and send the complete response.  The non-blocking Task
        wrapper is what enables instant interrupt handling.
        """
        try:
            # Extract last user utterance from transcript
            user_text = ""
            for turn in reversed(transcript):
                if turn.get("role") == "user":
                    user_text = turn.get("content", "")
                    break

            if not user_text:
                user_text = "Hola"

            # Config with thread_id for checkpointer persistence
            config = {"configurable": {"thread_id": cid}}

            # Input state: inject SYSTEM_PROMPT + user message
            input_state = {
                "messages": [SYSTEM_PROMPT, HumanMessage(content=user_text)],
                "call_id": cid,
            }

            # Run the graph (deterministic nodes — adapted from
            # astream_events to ainvoke since there's no LLM streaming)
            result = await graph.ainvoke(input_state, config=config)

            # Extract the last AI message as the agent reply
            agent_reply = ""
            for msg in reversed(result.get("messages", [])):
                if hasattr(msg, "content") and msg.type == "ai":
                    agent_reply = msg.content
                    break

            logger.info("Response — call_id=%s reply=%s", cid, agent_reply[:80])

            # Send content chunk to Retell
            if agent_reply:
                await ws.send_json({
                    "response_id": response_id,
                    "content": agent_reply,
                    "content_complete": False,
                })

            # Signal stream completion
            await ws.send_json({
                "response_id": response_id,
                "content": "",
                "content_complete": True,
            })

        except asyncio.CancelledError:
            # Interrupted by user — exit silently
            logger.info(
                "Generation cancelled — call_id=%s response_id=%s", cid, response_id
            )
            return

        except Exception:
            logger.exception("Generation error — call_id=%s", cid)

    # ---------------------------------------------------------------
    # Main receive loop
    # ---------------------------------------------------------------

    try:
        while True:
            raw = await ws.receive_text()
            data: dict = json.loads(raw)

            # Support both Retell's native "interaction_type" field
            # and the "event" + "type" wrapper format
            interaction_type = data.get("interaction_type", "")
            event = data.get("event", "")
            update_type = data.get("type", "")

            # ========= INTERACTION BEGIN / CALL DETAILS =========
            if interaction_type == "call_details" or event == "interaction_begin":
                logger.info("Call started — call_id=%s", call_id)

                initiator = data.get("initiator", "")
                if initiator == "agent":
                    # Agent-initiated call: launch greeting task
                    current_task = asyncio.create_task(
                        handle_generation([], 0, call_id)
                    )

            # ========= RESPONSE REQUIRED =========
            elif (
                interaction_type == "response_required"
                or (event == "interaction_update" and update_type == "response_required")
            ):
                # Step 1: Cancel running generation (non-blocking)
                if current_task and not current_task.done():
                    current_task.cancel()

                # Step 2: Launch new generation task
                transcript = data.get("transcript", [])
                response_id = data.get("response_id", 0)

                current_task = asyncio.create_task(
                    handle_generation(
                        transcript,
                        response_id,
                        call_id,
                    )
                )

            # ========= INTERRUPT =========
            elif (
                interaction_type == "update_only"
                or (event == "interaction_update" and update_type == "interrupt")
            ):
                # Immediate cancellation — do NOT send anything to the socket
                if current_task and not current_task.done():
                    current_task.cancel()
                logger.info("Interrupt received — call_id=%s", call_id)

            # ========= REMINDER REQUIRED =========
            elif interaction_type == "reminder_required":
                # Agent hasn't spoken for a while — re-trigger generation
                if current_task and not current_task.done():
                    current_task.cancel()

                response_id = data.get("response_id", 0)
                current_task = asyncio.create_task(
                    handle_generation(
                        data.get("transcript", []),
                        response_id,
                        call_id,
                    )
                )

            # ========= PING / PONG =========
            elif interaction_type == "ping_pong":
                await ws.send_json({
                    "interaction_type": "ping_pong",
                    "timestamp": data.get("timestamp", 0),
                })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected — call_id=%s", call_id)
    except Exception:
        logger.exception("WebSocket error — call_id=%s", call_id)
    finally:
        # Clean up any in-flight task
        if current_task and not current_task.done():
            current_task.cancel()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
