"""
WorldLens — Main Agent Entry Point
===================================
Day 1: Core agent with Gemini Realtime voice+vision over GetStream Edge.
Supports dual-mode operation (SignBridge / GuideLens) via AGENT_MODE env var.

Run:
    # Development (console mode — speaks through terminal)
    uv run main.py run

    # Server mode (API endpoint for frontend to create sessions)
    uv run main.py serve --host 0.0.0.0 --port 8000
"""

import logging
import os

from dotenv import load_dotenv

from vision_agents.core import Agent, AgentLauncher, Runner, ServeOptions, User
from vision_agents.core.events import (
    CallSessionParticipantJoinedEvent,
    CallSessionParticipantLeftEvent,
)
from vision_agents.core.llm.events import (
    RealtimeAgentSpeechTranscriptionEvent,
    RealtimeUserSpeechTranscriptionEvent,
)
from vision_agents.plugins import gemini, getstream

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("worldlens")

AGENT_MODE = os.getenv("AGENT_MODE", "guidelens")  # "signbridge" | "guidelens"

# ---------------------------------------------------------------------------
# System prompts per mode
# ---------------------------------------------------------------------------
SIGNBRIDGE_INSTRUCTIONS = """You are SignBridge — an advanced sign-language translation assistant.
You observe a user's video feed showing sign language gestures.
When you detect signing, describe the translated meaning clearly and concisely.
If the user speaks to you, respond helpfully with context about what you see.
Always be respectful, patient, and clear. Avoid jargon."""

GUIDELENS_INSTRUCTIONS = """You are GuideLens — a real-time environmental awareness assistant for visually impaired users.
You analyse the user's live camera feed and describe:
  • Nearby obstacles and hazards (poles, steps, vehicles, people approaching)
  • Text visible in the scene (signs, bus numbers, labels) — read them aloud
  • General scene layout when asked ("What is around me?")

Rules:
  - Be extremely concise — the user needs rapid, actionable info.
  - Prioritise safety: if something dangerous is approaching, say so IMMEDIATELY.
  - When the user asks for directions, use the get_walking_directions tool.
  - When the user asks "What did I see earlier?", use the search_memory tool.
  - Speak in natural, conversational sentences."""


def _get_instructions() -> str:
    return SIGNBRIDGE_INSTRUCTIONS if AGENT_MODE == "signbridge" else GUIDELENS_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
async def create_agent(**kwargs) -> Agent:
    """Create and configure a WorldLens agent instance."""
    instructions = _get_instructions()

    # --- Core Agent --------------------------------------------------------
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="WorldLens", id="worldlens-agent"),
        instructions=instructions,
        llm=gemini.Realtime(fps=5),  # 5 frames-per-second to Gemini Realtime
        processors=[],  # Day 2: YOLO / Moondream processors go here
    )

    # --- Event Logging (helpful during development) -------------------------
    @agent.events.subscribe
    async def on_user_speech(event: RealtimeUserSpeechTranscriptionEvent):
        logger.info("🎤 User: %s", event.text)

    @agent.events.subscribe
    async def on_agent_speech(event: RealtimeAgentSpeechTranscriptionEvent):
        logger.info("🤖 Agent: %s", event.text)

    @agent.events.subscribe
    async def on_participant_joined(event: CallSessionParticipantJoinedEvent):
        if event.participant.user.id != "worldlens-agent":
            logger.info("👤 Participant joined: %s", event.participant.user.name)

    @agent.events.subscribe
    async def on_participant_left(event: CallSessionParticipantLeftEvent):
        if event.participant.user.id != "worldlens-agent":
            logger.info("👋 Participant left: %s", event.participant.user.name)

    # --- MCP Tool stubs (wired up on Day 4) --------------------------------
    @agent.llm.register_function(
        name="get_walking_directions",
        description="Get step-by-step walking directions from current location to a destination using Google Maps.",
    )
    async def get_walking_directions(destination: str) -> dict:
        # Day 4: Wire to Google Maps Directions API
        return {
            "status": "stub",
            "message": f"Walking directions to '{destination}' will be available after Day 4 integration.",
        }

    @agent.llm.register_function(
        name="search_memory",
        description="Search the spatial memory database for previously seen objects or events.",
    )
    async def search_memory(query: str) -> dict:
        # Day 4: Wire to SQLite spatial_memory.db
        return {
            "status": "stub",
            "message": f"Memory search for '{query}' will be available after Day 4 integration.",
        }

    logger.info("Agent created in [%s] mode", AGENT_MODE.upper())
    return agent


# ---------------------------------------------------------------------------
# Call handler
# ---------------------------------------------------------------------------
async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Handle an agent joining a Stream call."""
    call = await agent.create_call(call_type, call_id)
    logger.info("Joining call %s/%s …", call_type, call_id)

    async with agent.join(call):
        # Greet the user based on mode
        if AGENT_MODE == "guidelens":
            await agent.simple_response(
                "Hello! I'm GuideLens, your real-time environmental assistant. "
                "Point your camera around and I'll describe what I see. "
                "Ask me anything about your surroundings."
            )
        else:
            await agent.simple_response(
                "Hello! I'm SignBridge, your sign-language translation assistant. "
                "Start signing whenever you're ready and I'll translate for you."
            )

        await agent.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    runner = Runner(
        AgentLauncher(
            create_agent=create_agent,
            join_call=join_call,
            max_concurrent_sessions=5,
            max_sessions_per_call=1,
            max_session_duration_seconds=1800,  # 30 min max
            agent_idle_timeout=120.0,  # 2 min alone → auto-close
        ),
        serve_options=ServeOptions(
            cors_allow_origins=["http://localhost:5173", "http://localhost:3000"],
            cors_allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            cors_allow_headers=["*"],
            cors_allow_credentials=True,
        ),
    )

    # Custom health endpoint
    @runner.fast_api.get("/mode")
    def get_mode():
        return {"mode": AGENT_MODE}

    runner.cli()
