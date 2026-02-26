"""
WorldLens — Main Agent Entry Point
===================================
Day 2: Core agent with Gemini Realtime voice+vision over GetStream Edge,
concurrent vision processors (SignBridge YOLO Pose / GuideLens YOLO Detection),
and mode-switching API.

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
from vision_agents.core.llm.events import (
    RealtimeAgentSpeechTranscriptionEvent,
    RealtimeUserSpeechTranscriptionEvent,
)
from vision_agents.plugins import gemini, getstream
from vision_agents.plugins.getstream import (
    CallSessionParticipantJoinedEvent,
    CallSessionParticipantLeftEvent,
)

from processors import (
    SignBridgeProcessor,
    SignDetectedEvent,
    GestureBufferEvent,
    SignTranslationEvent,
    GuideLensProcessor,
    ObjectDetectedEvent,
    HazardDetectedEvent,
    SceneSummaryEvent,
)

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
# System prompts per mode (updated for Day 2 processor awareness)
# ---------------------------------------------------------------------------
SIGNBRIDGE_INSTRUCTIONS = """You are SignBridge — an advanced sign-language translation assistant.

You observe a user's video feed which is processed by a YOLO Pose Estimation
pipeline that detects skeletal keypoints and recognises basic sign-language
gestures. The system extracts 17 COCO keypoints per person, tracks wrist
movements, and classifies gestures in real-time.

Behaviour:
  • When signing is detected, describe the gesture or its translated meaning.
  • If a gesture is classified (WAVE, RAISE-HAND, POINT, ACTIVE-SIGN), relay
    the meaning conversationally.
  • If the user speaks to you, respond helpfully.
  • You also see the raw video — use both the skeletal data and your own
    visual understanding to provide the best interpretation.

Always be respectful, patient, and clear. Avoid jargon."""

GUIDELENS_INSTRUCTIONS = """You are GuideLens — a real-time environmental awareness assistant for
visually impaired users.

You analyse the user's live camera feed which is processed by a YOLO Object
Detection pipeline. The system detects objects (people, vehicles, obstacles),
estimates their direction (left/centre/right) and distance (near/medium/far),
and tracks approaching objects via bounding-box growth rate.

Behaviour:
  • Nearby obstacles and hazards — announce them IMMEDIATELY with direction
    and distance (e.g. "Person approaching from the left, about 3 metres").
  • Text visible in the scene (signs, bus numbers, labels) — read them aloud.
  • General scene layout when asked ("What is around me?").
  • When the user asks for directions, use the get_walking_directions tool.
  • When the user asks "What did I see earlier?", use the search_memory tool.

Rules:
  - Be extremely concise — the user needs rapid, actionable info.
  - Prioritise safety above all else.
  - Speak in natural, conversational sentences."""


def _get_instructions() -> str:
    return (
        SIGNBRIDGE_INSTRUCTIONS
        if AGENT_MODE == "signbridge"
        else GUIDELENS_INSTRUCTIONS
    )


def _build_processors() -> list:
    """Instantiate the vision processors for the current mode."""
    if AGENT_MODE == "signbridge":
        logger.info("Building SignBridge processor (YOLO Pose)")
        return [
            SignBridgeProcessor(
                fps=10,
                conf_threshold=0.5,
                model_path="yolo11n-pose.pt",
                device="cpu",
                gesture_buffer_frames=30,
            )
        ]
    else:
        logger.info("Building GuideLens processor (YOLO Detection)")
        return [
            GuideLensProcessor(
                fps=5,
                conf_threshold=0.4,
                model_path="yolo11n.pt",
                device="cpu",
                scene_summary_interval=10.0,
            )
        ]


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
async def create_agent(**kwargs) -> Agent:
    """Create and configure a WorldLens agent instance."""
    instructions = _get_instructions()
    processors = _build_processors()

    # --- Core Agent --------------------------------------------------------
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="WorldLens", id="worldlens-agent"),
        instructions=instructions,
        llm=gemini.Realtime(fps=5),
        processors=processors,
    )

    # --- Event Logging (development) ----------------------------------------
    @agent.events.subscribe
    async def on_user_speech(event: RealtimeUserSpeechTranscriptionEvent):
        logger.info("🎤 User: %s", event.text)

    @agent.events.subscribe
    async def on_agent_speech(event: RealtimeAgentSpeechTranscriptionEvent):
        logger.info("🤖 Agent: %s", event.text)

    @agent.events.subscribe
    async def on_participant_joined(event: CallSessionParticipantJoinedEvent):
        if event.participant.user.id != "worldlens-agent":
            logger.info(
                "👤 Participant joined: %s", event.participant.user.name
            )

    @agent.events.subscribe
    async def on_participant_left(event: CallSessionParticipantLeftEvent):
        if event.participant.user.id != "worldlens-agent":
            logger.info(
                "👋 Participant left: %s", event.participant.user.name
            )

    # --- Processor event logging -------------------------------------------
    @agent.events.subscribe
    async def on_sign_detected(event: SignDetectedEvent):
        logger.info(
            "🤟 Sign detected: %d person(s), frame %d",
            event.num_persons,
            event.frame_number,
        )

    @agent.events.subscribe
    async def on_gesture_buffer(event: GestureBufferEvent):
        if event.raw_gloss:
            logger.info("🤟 Gesture classified: %s", event.raw_gloss)

    @agent.events.subscribe
    async def on_sign_translation(event: SignTranslationEvent):
        logger.info(
            "🤟 Translation: %s → %s",
            event.raw_gloss,
            event.translated_text,
        )

    @agent.events.subscribe
    async def on_object_detected(event: ObjectDetectedEvent):
        if event.objects:
            logger.debug(
                "👁️ Objects (frame %d): %s",
                event.frame_number,
                ", ".join(event.objects[:5]),
            )

    @agent.events.subscribe
    async def on_hazard_detected(event: HazardDetectedEvent):
        logger.warning(
            "⚠️ HAZARD: %s [%s, %s] — growth %.3f/s",
            event.hazard_type,
            event.distance_estimate,
            event.direction,
            event.growth_rate,
        )

    @agent.events.subscribe
    async def on_scene_summary(event: SceneSummaryEvent):
        logger.info("📸 %s", event.summary)

    # --- MCP Tool stubs (wired up on Day 4) --------------------------------
    @agent.llm.register_function(
        name="get_walking_directions",
        description="Get step-by-step walking directions from current location to a destination using Google Maps.",
    )
    async def get_walking_directions(destination: str) -> dict:
        return {
            "status": "stub",
            "message": f"Walking directions to '{destination}' will be available after Day 4 integration.",
        }

    @agent.llm.register_function(
        name="search_memory",
        description="Search the spatial memory database for previously seen objects or events.",
    )
    async def search_memory(query: str) -> dict:
        return {
            "status": "stub",
            "message": f"Memory search for '{query}' will be available after Day 4 integration.",
        }

    logger.info(
        "Agent created in [%s] mode with %d processor(s)",
        AGENT_MODE.upper(),
        len(processors),
    )
    return agent


# ---------------------------------------------------------------------------
# Call handler
# ---------------------------------------------------------------------------
async def join_call(
    agent: Agent, call_type: str, call_id: str, **kwargs
) -> None:
    """Handle an agent joining a Stream call."""
    call = await agent.create_call(call_type, call_id)
    logger.info("Joining call %s/%s …", call_type, call_id)

    async with agent.join(call):
        if AGENT_MODE == "guidelens":
            await agent.simple_response(
                "Hello! I'm GuideLens, your real-time environmental assistant. "
                "I'm running YOLO object detection on your camera feed to spot "
                "nearby obstacles and hazards. Point your camera around and "
                "I'll describe what I see."
            )
        else:
            await agent.simple_response(
                "Hello! I'm SignBridge, your sign-language translation assistant. "
                "I'm running YOLO pose estimation to track your hand and body "
                "movements. Start signing whenever you're ready!"
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
            max_session_duration_seconds=1800,
            agent_idle_timeout=120.0,
        ),
        serve_options=ServeOptions(
            cors_allow_origins=["http://localhost:5173", "http://localhost:3000"],
            cors_allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            cors_allow_headers=[
                "Content-Type",
                "Authorization",
                "Accept",
                "Origin",
                "X-Requested-With",
            ],
            cors_allow_credentials=True,
        ),
    )

    # ----- Custom API endpoints -----

    @runner.fast_api.get("/mode")
    def get_mode():
        """Get the current agent mode."""
        return {"mode": AGENT_MODE}

    @runner.fast_api.post("/switch-mode")
    def switch_mode():
        """
        Toggle between signbridge and guidelens modes.
        Takes effect on the next session creation.
        """
        global AGENT_MODE
        AGENT_MODE = "signbridge" if AGENT_MODE == "guidelens" else "guidelens"
        logger.info("Mode switched to [%s]", AGENT_MODE.upper())
        return {
            "mode": AGENT_MODE,
            "message": f"Mode switched to {AGENT_MODE}. New sessions will use this mode.",
        }

    @runner.fast_api.post("/set-mode/{mode}")
    def set_mode(mode: str):
        """Set a specific agent mode."""
        global AGENT_MODE
        if mode not in ("signbridge", "guidelens"):
            return {"error": f"Invalid mode: {mode}. Must be 'signbridge' or 'guidelens'."}
        AGENT_MODE = mode
        logger.info("Mode set to [%s]", AGENT_MODE.upper())
        return {"mode": AGENT_MODE}

    runner.cli()
