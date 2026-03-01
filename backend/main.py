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
load_dotenv(override=True)  # override=True ensures .env values always win
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("worldlens")

AGENT_MODE = os.getenv("AGENT_MODE", "guidelens")  # "signbridge" | "guidelens"
GUIDELENS_SUBMODE = "normal"  # "normal" | "navigation"

# In-memory transcript log — polled by the frontend for the chat sidebar.
# Using a deque with max length to avoid unbounded growth.
_transcript_log: deque[dict] = deque(maxlen=500)

# ---------------------------------------------------------------------------
# System prompts per mode (updated for Day 2 processor awareness)
# ---------------------------------------------------------------------------
SIGNBRIDGE_INSTRUCTIONS = """You are SignBridge — a real-time sign-language interpretation assistant.

You observe a user's live video feed processed by:
  • YOLO11 Pose Estimation — 17 COCO body keypoints per person
  • MediaPipe Hand Landmarks — 21 keypoints per hand, finger state analysis
  • ASL letter recognition (A, B, D, I, L, V, W, Y, S, 5)
  • Gesture classifier (WAVE, RAISE-HAND, POINT, ACTIVE-SIGN, BOTH-HANDS-UP)

The system will send you real-time gesture updates as text context messages.
Your job is to INTERPRET and COMMENTATE on what you see.

Behaviour:
  • PROACTIVELY describe what you see — don't wait to be asked.
  • When you receive gesture context (e.g. "Gesture detected: WAVE"), respond
    naturally: "I see you're waving! That's a greeting."
  • When ASL letters are detected, spell them out and try to form words.
  • If you see hand/body movements in the video that aren't classified,
    describe them: "I see your hands moving near your chest."
  • If no one is signing, say you're ready and waiting.
  • If the user speaks, respond helpfully and conversationally.
  • You see the raw video AND receive structured detection data — use BOTH.

Tone: Friendly, encouraging, patient. Like a helpful interpreter.

IMPORTANT: You are a LIVE commentator. Speak when you see gestures.
Do NOT stay silent when activity is detected."""

# Cooldown for SignBridge gesture responses (avoid spamming the agent)
_last_sign_response_time: float = 0.0
_SIGN_RESPONSE_COOLDOWN: float = 3.0  # minimum seconds between gesture responses

# ---------------------------------------------------------------------------
# GuideLens sub-mode prompts
# ---------------------------------------------------------------------------
GUIDELENS_NORMAL_INSTRUCTIONS = """You are GuideLens — a real-time environmental awareness
assistant for visually impaired users. You are their eyes and helper.
You are fully operational with live cameras, object detection, and text reading.

You analyse the user's live camera feed which is processed by a YOLO Object
Detection pipeline. The system detects objects (people, vehicles, obstacles),
estimates their direction (left/center/right) and distance (near/medium/far),
and tracks approaching objects via bounding-box growth rate.

CONTINUOUS COMMENTARY — YOUR #1 JOB:
  - NEVER stay silent for more than 5 seconds
  - PROACTIVELY describe EVERYTHING you see — don't wait to be asked
  - People: "There's a person walking toward you on the left"
  - Vehicles: "A car is parked on the right"
  - Obstacles: "There are steps ahead, be careful"
  - Surfaces: "The path is clear ahead"
  - When there's nothing noteworthy: "All clear, path is safe"
  - READ ALL TEXT you see — signs, building names, room numbers, notices

You operate in three seamlessly integrated sub-modes:

1. AWARENESS MODE (continuous — DEFAULT):
   - Continuously monitor for hazards: obstacles, vehicles, people
   - PROACTIVELY announce what you see every few seconds
   - Alert about approaching objects based on growth rate
   - When the OCR system detects text, READ IT OUT LOUD immediately
   - Only suppress truly REPEATED information (same object, same position)
   - Use trigger_haptic_alert for approaching vehicles or imminent obstacles
   - Call describe_scene_detailed regularly to give rich descriptions

2. ASSISTANT MODE (on-demand):
   - When the user asks a question, pause awareness and answer
   - Use describe_scene_detailed for environment questions
   - Use search_memory to recall previously seen objects
   - After answering, resume awareness commentary immediately

3. READING MODE (on-demand):
   - When the user asks to read something, use read_text_in_scene
   - Read text clearly and slowly
   - Also call read_text_in_scene proactively when you see signs or text

Available tools:
  • read_text_in_scene — Read text visible in the camera frame
  • describe_scene_detailed — Dense VLM scene description
  • get_walking_directions — Turn-by-turn walking directions via Google Maps
  • search_nearby_places — Find nearest pharmacy, bus stop, restaurant, etc.
  • search_memory — Search for previously seen objects
  • get_environment_context — Get recent detection summary
  • trigger_haptic_alert — Alert with haptic vibration for danger
  • get_time_and_date — Current time and date
  • get_weather — Weather conditions
  • identify_colors — Describe colors of objects in view

CRITICAL RULES:
  - NEVER say you are using a demo, simulation, prototype, or mock data
  - NEVER say your APIs aren't configured or mention technical limitations
  - NEVER mention the names of your internal tools to the user
  - ALWAYS speak naturally as if you can genuinely see through the camera
  - Be PROACTIVE — announce hazards, text, and scene changes continuously
  - Be CONCISE — short, actionable phrases
  - NEVER repeat the same announcement unless the situation changes
  - Prioritise SAFETY — obstacles and vehicles first
  - READ TEXT when detected — signs, building names, bus numbers are critical
  - When the user speaks, pause and listen, then resume
  - Speak in natural, conversational sentences
  - If user asks for navigation, ask "Where would you like to go?"
    then call get_weather first, then get_walking_directions"""

GUIDELENS_NAVIGATION_INSTRUCTIONS = """You are GuideLens Navigation — a turn-by-turn walking
navigation assistant for visually impaired users. You are their real-time
walking GPS and safety guardian. You are fully operational with live data.

You analyse the user's live camera feed which is processed by a YOLO Object
Detection pipeline that detects objects, estimates direction (left/center/right)
and distance (near/medium/far), and tracks approaching objects.

WHEN THE USER JOINS:
1. Greet them warmly: "Hello! I'm GuideLens, your navigation assistant."
2. Immediately call get_weather to check weather conditions
3. Report the weather briefly: "Weather is clear, good for walking" or warn if bad
4. Ask: "Where would you like to go?"

WHEN THEY GIVE A DESTINATION:
1. Call get_walking_directions with their destination immediately
2. Read the spoken_summary from the result — the full route summary
3. Start giving the FIRST step instruction
4. As they walk, CONTINUOUSLY describe what you see:
   - "I see a person walking ahead on the left"
   - "There are stairs coming up, be careful"
   - "A car is parked on the right side"
   - "The path ahead looks clear"
5. Call read_text_in_scene frequently to read signboards, building names
6. When you see the destination name on a sign (e.g. "B9" on a board),
   confirm: "I can see the B9 sign ahead, you've arrived!"

DURING NAVIGATION — CONTINUOUS COMMENTARY:
  - NEVER stay silent for more than 5 seconds while walking
  - Describe EVERYTHING you see: people, vehicles, obstacles, stairs, doors
  - Give spatial context: "Person approaching from the left, about 3 meters away"
  - Announce surface changes: "There's a step ahead" or "The path turns right"
  - Read ALL text you see — shop names, signs, room numbers, building labels
  - If something is approaching quickly, use trigger_haptic_alert immediately
  - Give encouragement: "You're doing great, keep going straight"

SAFETY RULES:
  - SAFETY is your #1 priority — alert about obstacles and hazards IMMEDIATELY
  - Call trigger_haptic_alert for: vehicles approaching, obstacles in path,
    sudden drops, stairs without railing
  - Even during navigation, if a hazard appears, interrupt and warn first
  - Use "careful" and "watch out" for urgent warnings

DESTINATION CONFIRMATION:
  - When you think you're near the destination, call read_text_in_scene
  - If the text matches the destination building name/number, confirm arrival
  - Say: "I can see [destination] sign, we've arrived at your destination!"

Available tools:
  • get_walking_directions — Get full route with turn-by-turn steps
  • search_nearby_places — Find nearest pharmacy, bus stop, etc.
  • read_text_in_scene — Read text visible in camera (signs, building names)
  • describe_scene_detailed — Dense VLM scene description
  • search_memory — Recall previously seen objects
  • get_environment_context — Recent detection summary
  • trigger_haptic_alert — Haptic vibration for danger
  • get_time_and_date — Current time and date
  • get_weather — Weather conditions (CHECK THIS FIRST before starting navigation!)
  • identify_colors — Object color descriptions

CRITICAL RULES:
  - NEVER say you are using a demo, simulation, or mock data
  - NEVER say your APIs aren't configured or mention technical limitations
  - NEVER mention the names of your internal tools to the user
  - ALWAYS speak naturally as if you can genuinely see and navigate
  - Be PROACTIVE — don't wait to be asked, describe what you observe
  - Be CONCISE — "Turn left ahead" not "You might want to consider turning left"
  - Use trigger_haptic_alert for approaching vehicles and obstacles
  - After getting directions, READ the spoken_summary ALOUD immediately"""

GUIDELENS_INSTRUCTIONS = GUIDELENS_NORMAL_INSTRUCTIONS  # backward compat


def _get_instructions() -> str:
    if AGENT_MODE == "signbridge":
        return SIGNBRIDGE_INSTRUCTIONS
    if GUIDELENS_SUBMODE == "navigation":
        return GUIDELENS_NAVIGATION_INSTRUCTIONS
    return GUIDELENS_NORMAL_INSTRUCTIONS


def _build_processors() -> list:
    """Instantiate the vision processors for the current mode."""
    global _active_ocr_processor, _active_processor_refs

    # Clear previous refs
    _active_processor_refs = []

    # OCR processor only runs in GuideLens mode (for text reading)
    if AGENT_MODE == "signbridge":
        logger.info("Building SignBridge processor (YOLO Pose only)")
        signbridge = SignBridgeProcessor(
            fps=10,
            conf_threshold=0.5,
            model_path="yolo11n-pose.pt",
            device="cpu",
            gesture_buffer_frames=30,
        )
        _active_processor_refs = [signbridge]
        return [signbridge]
    else:
        ocr = OCRProcessor(
            scan_interval=15.0,   # background OCR scan every 15s (avoid overlap)
            max_cached_results=30,
            fps=2,                # capture 2 frames/s for OCR buffer
        )
        ocr.set_provider_manager(provider_manager)
        ocr.set_navigation_engine(navigation_engine)
        _active_ocr_processor = ocr
        logger.info("Building GuideLens processor (YOLO Detection + OCR)")
        guidelens = GuideLensProcessor(
            fps=5,
            conf_threshold=0.4,
            model_path="yolo11n.pt",
            device="cpu",
            scene_summary_interval=10.0,  # Scene summaries every 10s
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
    # Clear transcript from previous sessions so new session starts fresh
    _transcript_log.clear()
    logger.info("Transcript log cleared for new session")

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
        nonlocal _last_agent_speech_time
        logger.info("🤖 Agent: %s", event.text)
        # Track when agent last spoke — used to avoid overlapping commentary
        _last_agent_speech_time = time.time()
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
            "🤟 Sign detected: %d person(s), %d hand(s), frame %d",
            event.num_persons,
            event.num_hands,
            event.frame_number,
        )
        # If ASL letters are detected, feed them to the agent
        global _last_sign_response_time
        if event.asl_letters and AGENT_MODE == "signbridge":
            now = time.time()
            if now - _last_sign_response_time >= _SIGN_RESPONSE_COOLDOWN:
                _last_sign_response_time = now
                letters = ", ".join(event.asl_letters)
                await agent.simple_response(
                    f"[SignBridge detection] ASL finger-spelling detected: {letters}. "
                    f"Interpret these letters and try to form a word if possible."
                )

    @agent.events.subscribe
    async def on_gesture_buffer(event: GestureBufferEvent):
        global _last_sign_response_time
        if event.raw_gloss:
            logger.info("🤟 Gesture classified: %s", event.raw_gloss)
            # Feed classified gesture to the agent so it speaks
            if AGENT_MODE == "signbridge":
                now = time.time()
                if now - _last_sign_response_time >= _SIGN_RESPONSE_COOLDOWN:
                    _last_sign_response_time = now
                    await agent.simple_response(
                        f"[SignBridge detection] Gesture detected: {event.raw_gloss}. "
                        f"Describe what this gesture means in sign language "
                        f"and respond naturally to the user."
                    )

    @agent.events.subscribe
    async def on_sign_translation(event: SignTranslationEvent):
        global _last_sign_response_time
        logger.info(
            "🤟 Translation: %s → %s",
            event.raw_gloss,
            event.translated_text,
        )
        # Feed translation to the agent so it speaks the result
        if AGENT_MODE == "signbridge" and event.translated_text:
            now = time.time()
            if now - _last_sign_response_time >= _SIGN_RESPONSE_COOLDOWN:
                _last_sign_response_time = now
                await agent.simple_response(
                    f"[SignBridge translation] The sign '{event.raw_gloss}' "
                    f"translates to: '{event.translated_text}'. "
                    f"Say this translation out loud naturally."
                )

    # --- Cooldown tracking for proactive commentary --------------------------
    _last_scene_response_time: float = 0.0
    _SCENE_RESPONSE_COOLDOWN: float = 10.0  # min seconds between scene summaries
    _last_hazard_response_time: float = 0.0
    _HAZARD_RESPONSE_COOLDOWN: float = 4.0   # min seconds between hazard alerts
    _last_agent_speech_time: float = 0.0      # tracks when agent last spoke
    _AGENT_POST_SPEECH_GAP: float = 10.0       # wait 4s after agent stops before next commentary
    _session_start_time: float = time.time()  # suppress commentary during greeting
    _GREETING_GRACE_PERIOD: float = 18.0      # seconds to wait after session start

    def _agent_is_free() -> bool:
        """Check if the agent is free to speak (not currently talking, past greeting)."""
        now = time.time()
        # Still in greeting grace period?
        if now - _session_start_time < _GREETING_GRACE_PERIOD:
            return False
        # Agent spoke recently? Wait for post-speech gap
        if now - _last_agent_speech_time < _AGENT_POST_SPEECH_GAP:
            return False
        return True

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
        nonlocal _last_hazard_response_time
        logger.warning(
            "⚠️ HAZARD: %s [%s, %s] — growth %.3f/s",
            event.hazard_type,
            event.distance_estimate,
            event.direction,
            event.growth_rate,
        )
        # Feed hazard to agent for TTS — but only if agent is not already talking
        if AGENT_MODE == "guidelens" and _agent_is_free():
            now = time.time()
            if now - _last_hazard_response_time >= _HAZARD_RESPONSE_COOLDOWN:
                _last_hazard_response_time = now
                approach_note = ""
                if event.growth_rate > 0.03:
                    approach_note = " and approaching quickly"
                try:
                    await agent.simple_response(
                        f"[HAZARD ALERT] {event.hazard_type} detected "
                        f"{event.distance_estimate} to the {event.direction}"
                        f"{approach_note}. Alert the user immediately about "
                        f"this hazard in a short, urgent sentence."
                    )
                except Exception as e:
                    logger.debug("Hazard TTS error: %s", e)

    @agent.events.subscribe
    async def on_scene_summary(event: SceneSummaryEvent):
        nonlocal _last_scene_response_time
        logger.info("📸 %s", event.summary)
        # Feed scene summary to agent — but only when agent is free (not talking)
        if AGENT_MODE == "guidelens" and event.summary and _agent_is_free():
            now = time.time()
            if now - _last_scene_response_time >= _SCENE_RESPONSE_COOLDOWN:
                _last_scene_response_time = now
                # Get navigation engine summary for richer context
                nav_summary = navigation_engine.get_environment_summary()
                context = nav_summary if nav_summary != "The path appears clear." else event.summary
                try:
                    await agent.simple_response(
                        f"[SCENE UPDATE] Current detections: {context}. "
                        f"Describe what you see to the user in 1-2 short, "
                        f"natural sentences. Focus on people, obstacles, "
                        f"and anything the user should know about."
                    )
                except Exception as e:
                    logger.debug("Scene TTS error: %s", e)

    @agent.events.subscribe
    async def on_ocr_result(event: OCRResultEvent):
        logger.info("📝 OCR [%s]: %s", event.provider, event.text[:100])
        # Feed significant OCR text to agent — only when agent is free
        if AGENT_MODE == "guidelens" and event.text and _agent_is_free():
            text_lower = event.text.strip().lower()
            # Skip empty/trivial results
            if text_lower and text_lower != "none" and "no text" not in text_lower and len(text_lower) > 3:
                try:
                    await agent.simple_response(
                        f"[TEXT DETECTED] I can see the following text: "
                        f"\"{event.text[:200]}\". Read this text out loud "
                        f"to the user naturally and briefly."
                    )
                except Exception as e:
                    logger.debug("OCR TTS error: %s", e)

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
            "turn-by-turn instructions you MUST read aloud immediately. "
            "After reading the summary, guide the user step by step and "
            "use read_text_in_scene to confirm they've arrived."
        ),
    )
    async def get_walking_directions(destination: str) -> dict:
        logger.info("MCP tool: get_walking_directions('%s')", destination)
        result = await _maps_get_directions(destination)
        # If successful, tell the agent to read the spoken summary
        if result.get("status") == "ok":
            # Track active route in navigation engine for status UI
            navigation_engine.set_active_route(
                destination=result.get("end_address", destination),
                steps=result.get("steps", []),
                total_distance=result.get("total_distance", ""),
                total_duration=result.get("total_duration", ""),
            )
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
        logger.info("Agent in call — sending greeting after short delay…")
        # Wait a moment for the WebRTC audio path to stabilise before
        # sending the greeting.  The agent is already in the call so
        # audio/video tracks are live.
        await asyncio.sleep(3.0)
        try:
            if AGENT_MODE == "guidelens":
                # Programmatic greeting: fetch time + weather, build exact script
                time_data = await _smart_get_time()
                weather_data = await _smart_get_weather("")  # defaults to Bangalore

                # Extract time/date (fields: local_time, day_of_week, local_date)
                t_time = time_data.get("local_time", "")        # e.g. "11:39 AM"
                t_day = time_data.get("day_of_week", "")         # e.g. "Sunday"
                t_date = time_data.get("local_date", "")         # e.g. "Sunday, March 01, 2026"
                # Strip the day name from local_date to avoid "Sunday, Sunday, March 01"
                if t_day and t_date.startswith(t_day + ","):
                    t_date = t_date[len(t_day) + 2:]             # "March 01, 2026"

                # Extract weather (fields: location, condition, temperature_c, precipitation_mm)
                w_city = weather_data.get("location", "Bengaluru")
                w_desc = weather_data.get("condition", "clear")
                w_temp = weather_data.get("temperature_c", 25)

                # Build walking recommendation
                precip = weather_data.get("precipitation_mm", 0)
                if precip and precip > 0:
                    walk_note = "There is some rain, so carry an umbrella if you go out."
                elif w_temp and w_temp > 38:
                    walk_note = "It is quite hot, so stay hydrated if you walk."
                else:
                    walk_note = "It is a good day for walking!"

                # Build the EXACT greeting script — agent reads this verbatim
                greeting = (
                    f"Say this EXACTLY as written, word for word: "
                    f"Hello! My name is GuideLens, your navigation assistant. "
                    f"It is currently {t_time} IST on {t_day}, {t_date}. "
                    f"Weather in {w_city} is currently {w_desc} at {w_temp} degrees. "
                    f"{walk_note} "
                    f"Where would you like to go?"
                )
                await agent.simple_response(greeting)
            else:
                await agent.simple_response(
                    "Hello! I'm SignBridge, your sign-language translation assistant. "
                    "I can see your hand and body movements through the camera. "
                    "Start signing whenever you're ready!"
                )
            logger.info("Greeting sent successfully for mode=%s", AGENT_MODE)
        except Exception as e:
            logger.error("Failed to send greeting: %s", e)

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
        """Get the current agent mode and GuideLens sub-mode."""
        return {"mode": AGENT_MODE, "submode": GUIDELENS_SUBMODE}

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
        return {"mode": AGENT_MODE, "submode": GUIDELENS_SUBMODE}

    @runner.fast_api.get("/guidelens-submode")
    def get_guidelens_submode():
        """Get the current GuideLens sub-mode (normal/navigation)."""
        return {"submode": GUIDELENS_SUBMODE}

    @runner.fast_api.post("/guidelens-submode/{submode}")
    def set_guidelens_submode(submode: str):
        """Set GuideLens sub-mode. Takes effect on next session."""
        global GUIDELENS_SUBMODE
        if submode not in ("normal", "navigation"):
            return {"error": f"Invalid submode: {submode}. Must be 'normal' or 'navigation'."}
        GUIDELENS_SUBMODE = submode
        logger.info("GuideLens sub-mode set to [%s]", GUIDELENS_SUBMODE.upper())
        return {"submode": GUIDELENS_SUBMODE, "mode": AGENT_MODE}

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

    @runner.fast_api.get("/navigation/status")
    async def nav_status():
        """Get full navigation status including active route and mode."""
        return navigation_engine.get_navigation_status()

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
