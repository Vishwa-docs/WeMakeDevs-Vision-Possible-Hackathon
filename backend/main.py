"""
WorldLens — Main Agent Entry Point
===================================
Day 4: Agentic tool calling, spatial memory (SQLite), navigation engine,
Google Maps integration, and multi-mode assistant capabilities.

Modes:
  - Navigation: Continuous obstacle detection + walking directions
  - Assistant: On-demand Q&A about the environment
  - Reading: OCR-focused text reading from signboards, books, phones

Run:
    # Development (console mode — speaks through terminal)
    uv run main.py run

    # Server mode (API endpoint for frontend to create sessions)
    uv run main.py serve --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
import os
import time
import uuid
from collections import deque

from dotenv import load_dotenv
from fastapi import Body

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
    OCRProcessor,
    OCRResultEvent,
    SceneDescriptionEvent,
)
from providers import provider_manager, ProviderID
from mcp_tools.spatial_memory import spatial_memory
from mcp_tools.maps_api import (
    get_walking_directions as _maps_get_directions,
    search_nearby_places as _maps_search_nearby,
    get_current_location_info as _maps_get_location,
)
from mcp_tools.navigation_engine import navigation_engine
from mcp_tools.smart_tools import (
    get_time_and_date as _smart_get_time,
    get_weather_info as _smart_get_weather,
    identify_color_in_scene as _smart_identify_color,
    trigger_emergency as _smart_emergency,
    get_device_status as _smart_device_status,
    get_emergency_log,
)

# Module-level reference to the active OCR processor (for API endpoints)
_active_ocr_processor: OCRProcessor | None = None
# Day 5: References to active processors for telemetry
_active_processor_refs: list = []

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

# In-memory transcript log — polled by the frontend for the chat sidebar.
# Using a deque with max length to avoid unbounded growth.
_transcript_log: deque[dict] = deque(maxlen=500)

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
visually impaired users. You are their eyes and navigator.

You analyse the user's live camera feed which is processed by a YOLO Object
Detection pipeline. The system detects objects (people, vehicles, obstacles),
estimates their direction (left/centre/right) and distance (near/medium/far),
and tracks approaching objects via bounding-box growth rate.

You operate in three seamlessly integrated sub-modes:

1. NAVIGATION MODE (continuous):
   - Continuously monitor for hazards: potholes, obstacles, vehicles, people
   - Only announce CHANGES in the environment — don't repeat yourself
   - Prioritise safety: nearby vehicles/obstacles first, then informational
   - When the user asks for directions, use get_walking_directions
   - When they ask "what's nearby?", use search_nearby_places

2. ASSISTANT MODE (on-demand):
   - When the user asks a question (e.g. "What color is that car?",
     "Who is Elon Musk?", "How's the weather?"), pause navigation and answer
   - Use describe_scene_detailed for environment questions
   - Use search_memory to recall previously seen objects
   - Use get_environment_context for recent detection history
   - After answering, automatically resume navigation awareness

3. READING MODE (on-demand):
   - When the user asks you to read something (sign, book, label, phone screen),
     use read_text_in_scene to read ALL visible text
   - Read text clearly and slowly for comprehension
   - If multiple text elements, read them in spatial order (top to bottom)

Available tools:
  • read_text_in_scene — Read text visible in the camera frame
  • describe_scene_detailed — Dense VLM scene description
  • get_walking_directions — Turn-by-turn walking directions via Google Maps
  • search_nearby_places — Find nearest pharmacy, bus stop, restaurant, etc.
  • search_memory — Search for previously seen objects ("Have you seen my keys?")
  • get_environment_context — Get recent detection summary for context
  • trigger_haptic_alert — Alert the user with a haptic vibration for approaching danger

Behaviour rules:
  - Be EXTREMELY concise in navigation mode — short, actionable phrases
  - NEVER repeat the same announcement unless the situation changes
  - Prioritise SAFETY above all else — obstacles and vehicles first
  - In assistant mode, give fuller answers but still be efficient
  - When there's no environmental change, stay SILENT
  - When the user speaks, pause announcements and listen
  - Speak in natural, conversational sentences
  - If the user asks about something you can't see, say so honestly"""


def _get_instructions() -> str:
    return (
        SIGNBRIDGE_INSTRUCTIONS
        if AGENT_MODE == "signbridge"
        else GUIDELENS_INSTRUCTIONS
    )


def _build_processors() -> list:
    """Instantiate the vision processors for the current mode."""
    global _active_ocr_processor, _active_processor_refs

    # Clear previous refs
    _active_processor_refs = []

    # OCR processor runs in both modes (captures frames for on-demand VLM)
    ocr = OCRProcessor(
        scan_interval=20.0,   # background OCR scan every 20s
        max_cached_results=30,
        fps=1,                # capture 1 frame/s for OCR buffer
    )
    ocr.set_provider_manager(provider_manager)
    _active_ocr_processor = ocr

    if AGENT_MODE == "signbridge":
        logger.info("Building SignBridge processor (YOLO Pose + OCR)")
        signbridge = SignBridgeProcessor(
            fps=10,
            conf_threshold=0.5,
            model_path="yolo11n-pose.pt",
            device="cpu",
            gesture_buffer_frames=30,
        )
        _active_processor_refs = [signbridge]
        return [signbridge, ocr]
    else:
        logger.info("Building GuideLens processor (YOLO Detection + OCR)")
        guidelens = GuideLensProcessor(
            fps=5,
            conf_threshold=0.4,
            model_path="yolo11n.pt",
            device="cpu",
            scene_summary_interval=10.0,
        )
        # Day 4: Wire spatial memory and navigation engine
        guidelens.set_spatial_memory(spatial_memory)
        guidelens.set_navigation_engine(navigation_engine)
        _active_processor_refs = [guidelens]
        return [guidelens, ocr]


# ---------------------------------------------------------------------------
# Transcript buffer — aggregates word-level transcript chunks into sentences
# ---------------------------------------------------------------------------
class _TranscriptAggregator:
    """Buffers rapid-fire transcript events into complete messages.

    The SDK's Gemini Realtime plugin emits a separate
    ``RealtimeAgentSpeechTranscriptionEvent`` for every word / short phrase.
    The SDK's built-in handler calls ``conversation.upsert_message`` with a
    **new UUID each time**, so each word becomes a separate Stream Chat
    message.

    This aggregator collects chunks and flushes them as a single chat
    message after a configurable silence window (default 1.5 s).
    """

    def __init__(self, flush_delay: float = 1.5):
        self._buffer: list[str] = []
        self._flush_delay = flush_delay
        self._flush_task: asyncio.Task | None = None
        self._message_id: str = str(uuid.uuid4())
        self._lock = asyncio.Lock()

    async def add(self, text: str, conversation, user_id: str, role: str):
        """Append a transcript chunk and (re)schedule the flush timer."""
        async with self._lock:
            self._buffer.append(text)

            # Cancel the previous flush timer — speaker is still talking
            if self._flush_task and not self._flush_task.done():
                self._flush_task.cancel()

            # Flush as a single upsert (streaming update, not completed yet)
            combined = "".join(self._buffer).strip()
            if conversation and combined:
                try:
                    await conversation.upsert_message(
                        message_id=self._message_id,
                        role=role,
                        user_id=user_id,
                        content=combined,
                        completed=False,
                        replace=True,
                    )
                except Exception:
                    pass  # non-critical

            # Schedule final flush
            self._flush_task = asyncio.create_task(
                self._flush(conversation, user_id, role)
            )

    async def _flush(self, conversation, user_id: str, role: str):
        """Wait for silence, then finalise the message and reset."""
        await asyncio.sleep(self._flush_delay)
        async with self._lock:
            combined = "".join(self._buffer).strip()
            if conversation and combined:
                try:
                    await conversation.upsert_message(
                        message_id=self._message_id,
                        role=role,
                        user_id=user_id,
                        content=combined,
                        completed=True,
                        replace=True,
                    )
                except Exception:
                    pass
                # Append to the in-memory transcript log for frontend polling
                speaker = "agent" if role == "assistant" else "user"
                _transcript_log.append(
                    {
                        "speaker": speaker,
                        "text": combined,
                        "timestamp": time.time() * 1000,
                    }
                )
            # Reset for the next utterance
            self._buffer.clear()
            self._message_id = str(uuid.uuid4())


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

    # Day 4: Initialise spatial memory (async SQLite)
    await spatial_memory.initialise()
    spatial_memory.set_session_id(f"session-{uuid.uuid4().hex[:8]}")
    logger.info("Spatial memory initialised with session: %s", spatial_memory._session_id)

    # --- Fix word-by-word chat: remove SDK built-in transcript→chat --------
    # The SDK subscribes its own handlers that call conversation.upsert_message
    # with a new UUID per transcript chunk, causing one chat message per word.
    # We remove those and replace them with buffered versions.
    _agent_type = "plugin.realtime_agent_speech_transcription"
    _user_type = "plugin.realtime_user_speech_transcription"
    for event_type in (_agent_type, _user_type):
        if event_type in agent.events._handlers:
            agent.events._handlers[event_type].clear()

    _agent_buf = _TranscriptAggregator(flush_delay=1.5)
    _user_buf = _TranscriptAggregator(flush_delay=1.0)

    # --- Replacement handlers with buffering -------------------------------
    @agent.events.subscribe
    async def on_user_speech(event: RealtimeUserSpeechTranscriptionEvent):
        logger.info("🎤 User: %s", event.text)
        # Suppress navigation announcements while the user is speaking
        navigation_engine.on_user_speech()
        if agent.conversation and event.text:
            uid = event.user_id() if hasattr(event, 'user_id') and callable(event.user_id) else "user"
            await _user_buf.add(
                event.text, agent.conversation, uid or "user", "user"
            )

    @agent.events.subscribe
    async def on_agent_speech(event: RealtimeAgentSpeechTranscriptionEvent):
        logger.info("🤖 Agent: %s", event.text)
        if agent.conversation and event.text:
            await _agent_buf.add(
                event.text, agent.conversation,
                agent.agent_user.id or "worldlens-agent", "assistant"
            )

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

    @agent.events.subscribe
    async def on_ocr_result(event: OCRResultEvent):
        logger.info("📝 OCR [%s]: %s", event.provider, event.text[:100])

    @agent.events.subscribe
    async def on_scene_description(event: SceneDescriptionEvent):
        logger.info("🏞️  Scene [%s]: %s", event.provider, event.description[:100])

    # --- OCR / VLM MCP Tools (Day 3) -----------------------------------------
    @agent.llm.register_function(
        name="read_text_in_scene",
        description=(
            "Read all text visible in the current camera frame. "
            "Use this when the user asks about signs, labels, bus numbers, "
            "notices, or any written text in their environment."
        ),
    )
    async def read_text_in_scene(prompt: str = "") -> dict:
        if _active_ocr_processor is None:
            return {"text": "", "error": "OCR processor not active"}
        return await _active_ocr_processor.read_text(prompt)

    @agent.llm.register_function(
        name="describe_scene_detailed",
        description=(
            "Generate a detailed, dense description of the current scene "
            "using an advanced Vision-Language Model (NVIDIA Cosmos / Gemini). "
            "Use when the user asks 'What is around me?', 'Describe this "
            "place', or needs a comprehensive overview of their environment."
        ),
    )
    async def describe_scene_detailed(prompt: str = "") -> dict:
        if _active_ocr_processor is None:
            return {"description": "", "error": "OCR processor not active"}
        return await _active_ocr_processor.describe_scene(prompt)

    # --- MCP Tool: Walking Directions (Day 4 — Live Google Maps) ----------
    @agent.llm.register_function(
        name="get_walking_directions",
        description=(
            "Get step-by-step walking directions from the user's current "
            "location to a destination using Google Maps. Also handles "
            "'nearest pharmacy', 'closest bus stop' type queries. Returns "
            "turn-by-turn instructions the agent should read aloud."
        ),
    )
    async def get_walking_directions(destination: str) -> dict:
        logger.info("MCP tool: get_walking_directions('%s')", destination)
        result = await _maps_get_directions(destination)
        # If successful, tell the agent to read the spoken summary
        if result.get("status") == "ok":
            return {
                "status": "ok",
                "spoken_summary": result.get("spoken_summary", ""),
                "steps": result.get("steps", []),
                "total_distance": result.get("total_distance", ""),
                "total_duration": result.get("total_duration", ""),
            }
        return result

    # --- MCP Tool: Search Nearby Places (Day 4) ---------------------------
    @agent.llm.register_function(
        name="search_nearby_places",
        description=(
            "Search for nearby places of a specific type (e.g., pharmacy, "
            "bus stop, restaurant, hospital, ATM). Returns a list of the "
            "closest places with distances."
        ),
    )
    async def search_nearby_places(place_type: str) -> dict:
        logger.info("MCP tool: search_nearby_places('%s')", place_type)
        return await _maps_search_nearby(place_type)

    # --- MCP Tool: Search Memory (Day 4 — Live SQLite) --------------------
    @agent.llm.register_function(
        name="search_memory",
        description=(
            "Search the spatial memory database for previously seen objects "
            "or events. Use when the user asks 'Have you seen my keys?', "
            "'What did I see earlier?', 'When did you last see a person?'. "
            "Returns matching detections with time-ago formatting."
        ),
    )
    async def search_memory(query: str) -> dict:
        logger.info("MCP tool: search_memory('%s')", query)
        results = await spatial_memory.search(query, limit=10)
        if not results:
            return {
                "status": "no_results",
                "message": f"I haven't seen any '{query}' in my memory.",
                "query": query,
            }
        return {
            "status": "ok",
            "query": query,
            "count": len(results),
            "results": results,
        }

    # --- MCP Tool: Environment Context (Day 4) ----------------------------
    @agent.llm.register_function(
        name="get_environment_context",
        description=(
            "Get a summary of recently detected objects in the environment. "
            "Use for answering questions about what's around the user or "
            "providing context for the current scene."
        ),
    )
    async def get_environment_context() -> dict:
        # Combine navigation engine state + spatial memory
        nav_summary = navigation_engine.get_environment_summary()
        mem_context = await spatial_memory.get_environment_context()
        mem_summary = await spatial_memory.get_summary()
        return {
            "status": "ok",
            "current_scene": nav_summary,
            "recent_history": mem_context,
            "memory_stats": {
                "total_detections": mem_summary.get("total_detections", 0),
                "unique_objects": mem_summary.get("unique_objects", 0),
                "recent_5min": mem_summary.get("recent_5min", 0),
            },
        }

    # --- MCP Tool: Haptic Alert (Day 5 — Enhanced) -------------------------
    @agent.llm.register_function(
        name="trigger_haptic_alert",
        description=(
            "Trigger a haptic/visual/audio alert on the user's device when an "
            "object is rapidly approaching or an immediate hazard is detected. "
            "Use sparingly — only for genuinely dangerous situations like "
            "a vehicle approaching, walking into an obstacle, or sudden danger. "
            "Parameters:\n"
            "  reason: Short description of the danger (e.g. 'Car approaching quickly from the left')\n"
            "  severity: 'critical' (imminent collision), 'warning' (approaching hazard), or 'caution' (be aware)\n"
            "  direction: 'left', 'center', or 'right' relative to user\n"
        ),
    )
    async def trigger_haptic_alert(
        reason: str,
        severity: str = "warning",
        direction: str = "center",
    ) -> dict:
        if severity not in ("critical", "warning", "caution"):
            severity = "warning"
        if direction not in ("left", "center", "right"):
            direction = "center"

        # Map severity to priority and sound type
        severity_map = {
            "critical": {"priority": 0, "sound": "siren", "duration_ms": 3000},
            "warning":  {"priority": 1, "sound": "beep",  "duration_ms": 2000},
            "caution":  {"priority": 2, "sound": "chime", "duration_ms": 1500},
        }
        meta = severity_map[severity]

        logger.warning(
            "🚨 HAPTIC ALERT [%s] dir=%s: %s", severity.upper(), direction, reason
        )

        alert_entry = {
            "text": reason,
            "priority": meta["priority"],
            "severity": severity,
            "type": "haptic_alert",
            "class": "agent_triggered",
            "direction": direction,
            "sound": meta["sound"],
            "duration_ms": meta["duration_ms"],
            "timestamp": time.time(),
        }
        navigation_engine._hazard_alerts.append(alert_entry)

        return {
            "status": "ok",
            "message": f"Haptic alert triggered: {reason}",
            "severity": severity,
            "direction": direction,
        }

    # --- MCP Tool: Time & Date (Day 5) ------------------------------------
    @agent.llm.register_function(
        name="get_time_and_date",
        description=(
            "Get the current local time, date, and day of the week. "
            "Use when the user asks 'What time is it?', 'What day is today?', "
            "or needs time-related information."
        ),
    )
    async def mcp_get_time() -> dict:
        logger.info("MCP tool: get_time_and_date()")
        return await _smart_get_time()

    # --- MCP Tool: Weather (Day 5) ----------------------------------------
    @agent.llm.register_function(
        name="get_weather",
        description=(
            "Get current weather conditions for a location. Includes "
            "temperature, humidity, wind speed, and conditions. "
            "Use when the user asks about weather, temperature, or if "
            "they should bring an umbrella."
        ),
    )
    async def mcp_get_weather(location: str = "") -> dict:
        logger.info("MCP tool: get_weather('%s')", location)
        return await _smart_get_weather(location)

    # --- MCP Tool: Color Identification (Day 5) ---------------------------
    @agent.llm.register_function(
        name="identify_colors",
        description=(
            "Identify and describe colors of objects visible in the camera. "
            "Use when the user asks 'What color is that car?', 'What color "
            "is this shirt?', or needs color-related help for accessibility."
        ),
    )
    async def mcp_identify_colors() -> dict:
        logger.info("MCP tool: identify_colors()")
        return await _smart_identify_color()

    # --- MCP Tool: Emergency Alert (Day 5) --------------------------------
    @agent.llm.register_function(
        name="emergency_alert",
        description=(
            "Trigger an emergency alert when the user is in danger or "
            "requests urgent help. Logs the emergency and in production "
            "would notify emergency contacts with GPS location. "
            "Use only for genuine emergencies — user says 'help', 'emergency', "
            "or appears to be in a dangerous situation."
        ),
    )
    async def mcp_emergency(reason: str, severity: str = "high") -> dict:
        logger.critical("MCP tool: emergency_alert('%s', '%s')", reason, severity)
        return await _smart_emergency(reason, severity)

    # --- MCP Tool: Device Status (Day 5) ----------------------------------
    @agent.llm.register_function(
        name="get_device_status",
        description=(
            "Check the device status including battery level, camera/mic "
            "status, and system uptime. Use when the user asks about "
            "battery, device health, or system status."
        ),
    )
    async def mcp_device_status() -> dict:
        logger.info("MCP tool: get_device_status()")
        return await _smart_device_status()

    logger.info(
        "Agent created in [%s] mode with %d processor(s) and 12 MCP tools",
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
    logger.info("[DEBUG] join_call begin call_type=%s call_id=%s", call_type, call_id)
    # Ensure the agent user is created on the edge *before* create_call,
    # because edge.create_call uses edge.agent_user_id which is set by
    # edge.create_user().  (create_user is idempotent — safe to call twice.)
    await agent.create_user()
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
    logger.info("[DEBUG] join_call end call_type=%s call_id=%s", call_type, call_id)


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

    # ----- Stream client for token generation -----
    from getstream import Stream as StreamClient
    from getstream.models import UserRequest

    _stream_client = StreamClient()

    # Upsert the agent user so Stream recognises it for call creation
    _stream_client.upsert_users(
        UserRequest(id="worldlens-agent", name="WorldLens", role="admin")
    )
    logger.info("Stream agent user upserted: worldlens-agent")

    # ----- Custom API endpoints -----

    @runner.fast_api.get("/token")
    async def get_user_token(user_id: str):
        """Generate a Stream user token for the frontend client.

        Also upserts the user so Stream's server-side auth recognises them
        when they join or create calls.
        """
        logger.info("[DEBUG] /token requested user_id=%s", user_id)
        # Upsert user into Stream so they exist before joining a call
        _stream_client.upsert_users(
            UserRequest(id=user_id, name=user_id, role="user")
        )
        token = _stream_client.create_token(user_id)
        logger.info("[DEBUG] /token issued user_id=%s token_len=%d", user_id, len(token or ""))
        return {"token": token, "user_id": user_id}

    @runner.fast_api.get("/stream-config")
    def get_stream_config():
        """Expose Stream config needed by the frontend client.

        Keeps backend as the single source of truth so frontend env drift
        (wrong or missing VITE_STREAM_API_KEY) does not break joins.
        """
        logger.info("[DEBUG] /stream-config requested")
        return {
            "api_key": os.getenv("STREAM_API_KEY", ""),
        }

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

    # ----- Provider management endpoints -----

    @runner.fast_api.get("/providers")
    async def get_providers():
        """Get status of all VLM providers and the current fallback chain."""
        health = await provider_manager.check_all_providers()
        status = provider_manager.get_status()
        status["health"] = health
        return status

    @runner.fast_api.post("/providers/preferred/{provider_id}")
    def set_preferred_provider(provider_id: str):
        """Set the user's preferred VLM provider."""
        ok = provider_manager.set_preferred(provider_id)
        if not ok:
            valid = [p.value for p in ProviderID]
            return {
                "error": f"Unknown provider: {provider_id}",
                "valid_providers": valid,
            }
        return provider_manager.get_status()

    @runner.fast_api.get("/providers/fallback-events")
    def get_fallback_events():
        """Poll for fallback events (for frontend toast notifications)."""
        events = provider_manager.pop_fallback_events()
        return {"events": events}

    @runner.fast_api.get("/transcript")
    def get_transcript(since: float = 0):
        """Get transcript entries, optionally filtering to entries after *since* (ms epoch)."""
        if since > 0:
            entries = [e for e in _transcript_log if e["timestamp"] > since]
        else:
            entries = list(_transcript_log)
        return {"entries": entries}

    @runner.fast_api.delete("/transcript")
    def clear_transcript():
        """Clear the transcript log (called when a session ends)."""
        _transcript_log.clear()
        return {"ok": True}

    # ----- OCR / VLM endpoints (Day 3) -----

    @runner.fast_api.get("/ocr-results")
    def get_ocr_results(since: float = 0, limit: int = 10):
        """Get cached OCR results for frontend overlay. Optionally filter by timestamp."""
        if _active_ocr_processor is None:
            return {"results": [], "available": False}
        results = _active_ocr_processor.get_recent_results(since=since, limit=limit)
        return {"results": results, "available": True}

    @runner.fast_api.post("/ocr/read")
    async def trigger_ocr_read(prompt: str = Body(default="")):
        """Manually trigger an OCR read of the current frame."""
        if _active_ocr_processor is None:
            return {"error": "OCR processor not active"}
        result = await _active_ocr_processor.read_text(prompt)
        return result

    @runner.fast_api.post("/ocr/describe")
    async def trigger_scene_describe(prompt: str = Body(default="")):
        """Manually trigger a detailed scene description."""
        if _active_ocr_processor is None:
            return {"error": "OCR processor not active"}
        result = await _active_ocr_processor.describe_scene(prompt)
        return result

    # ------------------------------------------------------------------
    # Day 4: Memory & Navigation API endpoints
    # ------------------------------------------------------------------

    @runner.fast_api.get("/memory/search")
    async def memory_search(q: str = "", limit: int = 10):
        """Search spatial memory for previously detected objects."""
        if not q:
            return {"error": "Missing query parameter 'q'"}
        results = await spatial_memory.search(q, limit=limit)
        return {"query": q, "count": len(results), "results": results}

    @runner.fast_api.get("/memory/summary")
    async def memory_summary():
        """Get aggregate memory statistics."""
        return await spatial_memory.get_summary()

    @runner.fast_api.get("/memory/recent")
    async def memory_recent(limit: int = 20):
        """Get most recent detections."""
        results = await spatial_memory.get_recent(limit=limit)
        return {"count": len(results), "results": results}

    @runner.fast_api.get("/memory/context")
    async def memory_context():
        """Get natural-language environment context from memory."""
        return {"context": await spatial_memory.get_environment_context()}

    @runner.fast_api.get("/navigation/summary")
    async def nav_summary():
        """Get current navigation engine environment summary."""
        return {"summary": navigation_engine.get_environment_summary()}

    @runner.fast_api.get("/navigation/hazards")
    async def nav_hazards(since: float = 0):
        """Get list of active hazard alerts, optionally since a timestamp."""
        return {"hazards": navigation_engine.get_hazard_alerts(since=since)}

    @runner.fast_api.get("/navigation/hazards/poll")
    async def nav_hazards_poll(since: float = 0):
        """Poll and consume hazard alerts (marks them as read)."""
        alerts = navigation_engine.pop_hazard_alerts(since=since)
        return {"hazards": alerts, "count": len(alerts)}

    @runner.fast_api.post("/navigation/assistant")
    async def nav_assistant_toggle(activate: bool = Body(default=True)):
        """Activate or deactivate assistant mode (pauses navigation announcements)."""
        if activate:
            navigation_engine.activate_assistant()
        else:
            navigation_engine.deactivate_assistant()
        return {"assistant_mode": activate}

    # ------------------------------------------------------------------
    # Day 5: Smart Tools API endpoints
    # ------------------------------------------------------------------

    @runner.fast_api.get("/time")
    async def api_get_time():
        """Get current time and date."""
        return await _smart_get_time()

    @runner.fast_api.get("/weather")
    async def api_get_weather(location: str = ""):
        """Get weather for a location."""
        return await _smart_get_weather(location)

    @runner.fast_api.get("/device-status")
    async def api_device_status():
        """Get device status info."""
        return await _smart_device_status()

    @runner.fast_api.get("/emergencies")
    def api_get_emergencies():
        """Get emergency alert log."""
        return {"emergencies": get_emergency_log()}

    @runner.fast_api.post("/emergency")
    async def api_trigger_emergency(reason: str = Body(default="User triggered"), severity: str = Body(default="high")):
        """Manually trigger an emergency alert."""
        return await _smart_emergency(reason, severity)

    # ------------------------------------------------------------------
    # Day 5: Real Telemetry Metrics Endpoint
    # ------------------------------------------------------------------

    # Module-level telemetry state — tracks session-level metrics
    _telemetry_start = time.time()
    _session_count = 0

    @runner.fast_api.get("/telemetry")
    async def get_telemetry():
        """
        Real telemetry metrics aggregated from all active processors,
        providers, spatial memory, and navigation engine.
        """
        # --- Processor metrics ---
        processor_metrics = []
        if _active_ocr_processor and hasattr(_active_ocr_processor, "get_telemetry"):
            processor_metrics.append(_active_ocr_processor.get_telemetry())

        # We need to inspect the current agent's processors.
        # The processors are created per-session in _build_processors(), and
        # we don't have a direct reference to the GuideLens/SignBridge instance
        # from here. But we can store them at creation time.
        for proc_ref in _active_processor_refs:
            if hasattr(proc_ref, "get_telemetry"):
                processor_metrics.append(proc_ref.get_telemetry())

        # --- Provider metrics ---
        provider_status = provider_manager.get_status()

        # --- Memory metrics ---
        try:
            mem_summary = await spatial_memory.get_summary()
        except Exception:
            mem_summary = {}

        # --- Navigation metrics ---
        nav_summary_text = navigation_engine.get_environment_summary()
        hazard_count = len(navigation_engine.get_hazard_alerts())

        # --- Aggregate ---
        total_frames = sum(
            p.get("frames_processed", 0) for p in processor_metrics
        )
        avg_inference = 0.0
        inference_procs = [
            p for p in processor_metrics
            if p.get("avg_inference_ms", 0) > 0
        ]
        if inference_procs:
            avg_inference = sum(
                p["avg_inference_ms"] for p in inference_procs
            ) / len(inference_procs)

        return {
            "mode": AGENT_MODE,
            "uptime_seconds": round(time.time() - _telemetry_start, 1),
            "processors": processor_metrics,
            "processor_count": len(processor_metrics),
            "aggregate": {
                "total_frames_processed": total_frames,
                "avg_inference_ms": round(avg_inference, 1),
                "total_objects_detected": sum(
                    p.get("total_objects_detected", 0) for p in processor_metrics
                ),
                "total_hazards_detected": sum(
                    p.get("total_hazards_detected", 0) for p in processor_metrics
                ),
                "total_gestures_detected": sum(
                    p.get("total_gestures_detected", 0) for p in processor_metrics
                ),
                "total_ocr_calls": sum(
                    p.get("total_ocr_calls", 0) for p in processor_metrics
                ),
            },
            "providers": {
                "preferred": provider_status.get("preferred", ""),
                "chain": provider_status.get("fallback_chain", []),
                "stats": {
                    pid: {
                        "calls": info.get("total_calls", 0),
                        "errors": info.get("total_errors", 0),
                        "available": info.get("available", False),
                    }
                    for pid, info in provider_status.get("providers", {}).items()
                },
            },
            "memory": {
                "total_detections": mem_summary.get("total_detections", 0),
                "unique_objects": mem_summary.get("unique_objects", 0),
                "recent_5min": mem_summary.get("recent_5min", 0),
            },
            "navigation": {
                "mode": navigation_engine.mode,
                "scene_summary": nav_summary_text,
                "pending_hazards": hazard_count,
            },
        }

    runner.cli()
