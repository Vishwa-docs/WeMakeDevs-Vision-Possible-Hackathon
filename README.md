# 🌍 WorldLens

**An autonomous, dual-mode assistive vision platform for accessibility.**

> Built for the WeMakeDevs Vision Possible Hackathon (Feb 2026)

WorldLens leverages the **Vision Agents SDK**, **Gemini 2.5 Flash Realtime**, **GetStream Edge Network**, **Google MediaPipe**, and **multi-provider VLM fallback** to provide two modes of real-time assistance:

- **🤟 SignBridge** — Real-time sign language translation via YOLO11 Pose + MediaPipe Hand Landmarks + HuggingFace NLP
- **👁️ GuideLens** — Environmental awareness for visually impaired users via YOLO11 Detection + MCP tool calling (Maps, Spatial Memory)

> NOTE: Architecture and Idea was designed by me. Used AI for code generation (Perplexity + GitHub Copilot) as a coding and refinement agent.
> NOTE : SignBridge mode was developed till prototype level for sign language translation. The goal was to deploy this as an app with 2 personas : A differently-abled user who uses sign language and another user who may not know sign language. This app would act as a bridge between the two. However, due to time contraints, I dropped this part.
> In the future, I hope to develop and add some features I was planning, namely **lip reading to speech** and **alerts when someone calls out to the user using vibrations**.

---

## Architecture

```
M5StickC Camera → Python Relay → Stream Edge (WebRTC) → Vision Agents Backend
                                                              │
                                    ┌─────────────────────────┼───────────────────┐
                                    │                         │                   │
                              YOLO Pose +             YOLO Detection        Gemini 2.5 Flash
                              MediaPipe Hands         (GuideLens)           Realtime
                              (SignBridge)                  │                (Voice + Vision)
                                    │                      │                     │
                              HuggingFace NLP       OCR Processor          MCP Tools
                              (Gloss→English)       (Multi-VLM)            (Maps / Memory)
                                    │                      . │                     │
                                    └──────────────────────┼─────────────────────┘
                                                           │
                                                  React 19 Frontend
                                               (Stream Video SDK + 3D Avatar)
```

### SignBridge Vision Pipeline
```
Camera Frame → YOLO11 Pose (17 keypoints) → MediaPipe Hands (21 landmarks/hand)
                    │                              │
              Skeleton Overlay              Finger State Analysis
              Gesture Classifier            ASL Letter Recognition
                    │                              │
                    └──────── Merged Detection ────┘
                                    │
                         SignDetectedEvent → Gemini Realtime → Voice Response
```

### GuideLens Vision Pipeline
```
Camera Frame → YOLO11 Detection (80 classes) → Hazard Filter
                                                     │
                                          BboxTracker (approach speed)
                                          Direction + Distance estimation
                                                     │
                                          HazardDetectedEvent → Haptic Alert
                                          SceneSummaryEvent   → Voice Warning
```

---

## Quick Start

### Prerequisites

- Python 3.12+ with [uv](https://docs.astral.sh/uv/)
- Node.js 18+ (recommended: 20 LTS)
- API keys (see below)

### Backend

```bash
cd backend

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your API keys (see "API Keys" section below)

# Run in development mode
uv run main.py run

# Run as server (for frontend)
uv run main.py serve --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Configure environment
cp .env.example .env
# Set VITE_STREAM_API_KEY and VITE_BACKEND_URL

# Start dev server
npm run dev
# Opens at http://localhost:5173
```

> See [backend/README.md](backend/README.md) and [frontend/README.md](frontend/README.md) for detailed setup guides.

---

## API Keys

### Required

| Variable | Service | How to Get |
|----------|---------|------------|
| `STREAM_API_KEY` | GetStream WebRTC | [getstream.io/dashboard](https://getstream.io/dashboard/) |
| `STREAM_API_SECRET` | GetStream WebRTC | Same dashboard as above |
| `GOOGLE_API_KEY` | Gemini 2.5 Flash Realtime | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |

### Recommended

| Variable | Service | How to Get |
|----------|---------|------------|
| `MAPS_API_KEY` | Google Maps (Directions, Geocoding, Places) | [console.cloud.google.com](https://console.cloud.google.com/apis/credentials) — enable Directions, Geocoding, Places, Geolocation APIs |
| `HF_API_TOKEN` | HuggingFace (gloss→English NLP) | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

### Optional (VLM Fallback Chain)

| Variable | Service |
|----------|---------|
| `NGC_API_KEY` | NVIDIA NGC (Cosmos VLM) |
| `XAI_API_KEY` | xAI Grok VLM |
| `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_KEY` / `AZURE_OPENAI_DEPLOYMENT` | Azure OpenAI |

> **Google Maps detailed setup:** See the step-by-step guide in [backend/README.md](backend/README.md#google-maps-api-detailed-setup).

---

## Vision Agents SDK Integration

WorldLens is built on top of the **Vision Agents SDK** (`vision-agents>=0.3.7`). Every core capability — agent lifecycle, LLM communication, video processing, event-driven architecture, tool calling, and WebRTC transport — uses the SDK's classes and APIs.

### VisionAgents / Stream SDK Classes & Where They're Used

| SDK Class / Module | Import | Where Used | Purpose |
|---|---|---|---|
| `Agent` | `vision_agents.core.Agent` | `main.py` → `create_agent()` | Core agent instance — holds LLM, processors, event bus, and conversation state |
| `AgentLauncher` | `vision_agents.core.AgentLauncher` | `main.py` → `__main__` | Manages agent lifecycle: creation, call joining, concurrency limits, idle timeouts |
| `Runner` | `vision_agents.core.Runner` | `main.py` → `__main__` | Top-level entry point — launches the agent as a FastAPI server with `serve` or standalone with `run` |
| `ServeOptions` | `vision_agents.core.ServeOptions` | `main.py` → `__main__` | Configures CORS, allowed origins/methods for the FastAPI HTTP server |
| `User` | `vision_agents.core.User` | `main.py` → `create_agent()` | Defines the agent's identity on the Stream edge (`WorldLens / worldlens-agent`) |
| `gemini.Realtime` | `vision_agents.plugins.gemini` | `main.py` → `create_agent()` | LLM backend — Gemini 2.5 Flash Realtime for speech-to-speech reasoning at 5 FPS |
| `getstream.Edge` | `vision_agents.plugins.getstream` | `main.py` → `create_agent()` | WebRTC transport — connects agent to GetStream's Edge Network for real-time video/audio |
| `VideoProcessorPublisher` | `vision_agents.core.processors` | `guidelens_processor.py`, `signbridge_processor.py` | Base class for video processors that can publish annotated frames back to the call |
| `VideoProcessor` | `vision_agents.core.processors` | `ocr_processor.py` | Base class for passive processors that read frames without publishing overlays |
| `VideoForwarder` | `vision_agents.core.utils.video_forwarder` | All 3 processors | Handles writing annotated frames back to the WebRTC video track |
| `QueuedVideoTrack` | `vision_agents.core.utils.video_track` | `guidelens_processor.py`, `signbridge_processor.py` | Thread-safe video track queue for non-blocking frame publishing |
| `BaseEvent` | `vision_agents.core.events` | All custom events | Base class for all custom events (`SignDetectedEvent`, `HazardDetectedEvent`, `OCRResultEvent`, etc.) |
| `agent.events.subscribe` | Event system | `main.py` → 13 event handlers | Decorator to subscribe to SDK and custom events (participant join/leave, speech transcription, processor events) |
| `agent.llm.register_function` | MCP / function calling | `main.py` → 12 MCP tools | Registers callable tools that Gemini can invoke during conversation (navigation, memory, weather, etc.) |
| `agent.simple_response()` | Agent API | `main.py` → `join_call()` + event handlers | Sends a text prompt to the LLM for immediate spoken response (greetings, sign translations) |
| `agent.create_user()` / `agent.create_call()` / `agent.join()` | Agent API | `main.py` → `join_call()` | Call lifecycle — creates the agent user on Stream, creates/joins a WebRTC call, and starts processing |
| `agent.conversation` | Conversation API | `main.py` → transcript handlers | Access to the conversation/chat state for upserting buffered transcript messages |
| `agent.finish()` | Agent API | `main.py` → `join_call()` | Signals the agent to cleanly exit the call after processing ends |
| `RealtimeUserSpeechTranscriptionEvent` | `vision_agents.core.llm.events` | `main.py` → `on_user_speech()` | SDK event fired when the user's speech is transcribed in real-time |
| `RealtimeAgentSpeechTranscriptionEvent` | `vision_agents.core.llm.events` | `main.py` → `on_agent_speech()` | SDK event fired when the agent's spoken response is transcribed |
| `CallSessionParticipantJoinedEvent` | `vision_agents.plugins.getstream` | `main.py` → `on_participant_joined()` | SDK event fired when a user joins the WebRTC call |
| `CallSessionParticipantLeftEvent` | `vision_agents.plugins.getstream` | `main.py` → `on_participant_left()` | SDK event fired when a user leaves the WebRTC call |

### How the SDK Powers Each Feature

1. **Real-time Voice Conversation** — `gemini.Realtime(fps=5)` provides speech-to-speech: the user talks, Gemini processes both audio and video frames, and speaks back. The SDK handles the entire audio pipeline.

2. **WebRTC Video Transport** — `getstream.Edge()` manages the WebRTC connection. The camera feed (from M5StickC or webcam) flows through Stream's edge network. The SDK's `VideoForwarder` and `QueuedVideoTrack` allow processors to send annotated frames (bounding boxes, skeleton overlays) back to the frontend.

3. **Video Processors** — All three processors extend the SDK's `VideoProcessorPublisher` or `VideoProcessor` base classes:
   - **SignBridgeProcessor** — Extends `VideoProcessorPublisher`. Runs YOLO11 Pose + MediaPipe Hand Landmarks on each frame, publishes skeleton overlay frames via `VideoForwarder`.
   - **GuideLensProcessor** — Extends `VideoProcessorPublisher`. Runs YOLO11 Detection, draws bounding boxes with hazard colors, publishes annotated frames.
   - **OCRProcessor** — Extends `VideoProcessor`. Captures frames passively for VLM-based text reading without publishing overlays.

4. **Custom Event System** — All processor outputs use the SDK's `BaseEvent` class. Events like `HazardDetectedEvent`, `SignDetectedEvent`, `SceneSummaryEvent` are published via the SDK event bus and consumed by handlers registered with `@agent.events.subscribe`.

5. **MCP Tool Calling** — 12 tools are registered via `@agent.llm.register_function()`. When Gemini determines the user needs directions, weather, or memory search, it calls these tools and receives structured JSON responses. This is the SDK's implementation of the Model Context Protocol.

6. **Agent Lifecycle** — `Runner` → `AgentLauncher` → `create_agent()` → `join_call()` → `agent.join(call)` → `agent.finish()`. The SDK manages concurrency (max 5 sessions), idle timeouts (120s), and session duration limits (30 min).

---

## Agents & MCP Tools

### The WorldLens Agent

WorldLens runs a single **agentic AI** — a Vision Agents `Agent` instance configured with:

- **LLM**: Gemini 2.5 Flash Realtime (speech-to-speech at 5 FPS)
- **Transport**: GetStream Edge (WebRTC)
- **System Instructions**: ~200-line dynamic prompt covering GuideLens navigation behavior, hazard response protocols, MCP tool usage guidelines, and SignBridge translation rules
- **Processors**: Up to 3 video processors depending on mode (GuideLens detector, SignBridge pose, OCR)
- **MCP Tools**: 12 registered functions the LLM can call autonomously

The agent joins a WebRTC call, receives live video + audio, processes frames through YOLO models, and converses with the user via Gemini — all orchestrated by the Vision Agents SDK.

### MCP Tools (12 Total)

All tools are registered via `@agent.llm.register_function()` — the SDK's implementation of the **Model Context Protocol**. Gemini decides when to call each tool based on conversation context.

| # | Tool Name | Module | What It Does | When the LLM Calls It |
|---|-----------|--------|-------------|----------------------|
| 1 | `read_text_in_scene` | `ocr_processor.py` | Captures the current camera frame and sends it to a VLM (Gemini/Grok/Azure/NVIDIA/HF fallback chain) to extract all visible text | User asks: *"What does that sign say?"*, *"Read the bus number"*, *"What's on that notice?"* |
| 2 | `describe_scene_detailed` | `ocr_processor.py` | Sends the current frame to a VLM for a dense, comprehensive scene description | User asks: *"What's around me?"*, *"Describe this place"*, *"What do you see?"* |
| 3 | `get_walking_directions` | `maps_api.py` | Fetches step-by-step walking directions from Google Maps Directions API. Handles "nearest X" queries by first resolving via Places API. Includes geolocation fallback. | User asks: *"How do I get to the train station?"*, *"Navigate to the nearest pharmacy"* |
| 4 | `search_nearby_places` | `maps_api.py` | Searches Google Maps Places API for nearby locations of a given type (pharmacy, bus stop, ATM, hospital, restaurant) | User asks: *"Where's the nearest pharmacy?"*, *"Find a bus stop nearby"* |
| 5 | `search_memory` | `spatial_memory.py` | Queries the async SQLite spatial memory database for previously detected objects, with time-ago formatting | User asks: *"Have you seen my keys?"*, *"When did you last see a person?"*, *"What did I see earlier?"* |
| 6 | `get_environment_context` | `spatial_memory.py` + `navigation_engine.py` | Combines current navigation engine state with spatial memory history to provide a holistic environment summary | User asks: *"What's been happening around me?"*, *"Give me context"* |
| 7 | `trigger_haptic_alert` | Built-in (`main.py`) | Triggers a visual + audio + haptic alert on the user's device with configurable severity (`critical`/`warning`/`caution`) and direction (`left`/`center`/`right`) | LLM detects imminent danger: *vehicle approaching*, *user walking toward obstacle*, *sudden hazard* |
| 8 | `get_time_and_date` | `smart_tools.py` | Returns current local time, date, and day of the week | User asks: *"What time is it?"*, *"What day is today?"* |
| 9 | `get_weather` | `smart_tools.py` | Fetches current weather from Open-Meteo API (free, no key required) — temperature, humidity, wind, conditions | User asks: *"What's the weather like?"*, *"Should I bring an umbrella?"*, *"Is it cold outside?"* |
| 10 | `identify_colors` | `smart_tools.py` | Describes colors of objects visible in the camera (uses VLM for color analysis) | User asks: *"What color is that car?"*, *"What color is this shirt?"* |
| 11 | `emergency_alert` | `smart_tools.py` | Logs an emergency event and (in production) would notify emergency contacts with GPS location | User says: *"Help!"*, *"Emergency!"*, or appears to be in danger |
| 12 | `get_device_status` | `smart_tools.py` | Returns device info — battery level, camera/mic status, system uptime | User asks: *"How's my battery?"*, *"Is the camera working?"* |

### Supporting MCP Infrastructure

These modules power the MCP tools but aren't directly callable by the LLM:

| Module | Purpose |
|--------|---------|
| `navigation_engine.py` | Priority-based hazard announcement queue, route step tracking, environment state machine (`idle` → `navigation` → `assistant` → `reading`), dedup cooldown to prevent repetitive warnings |
| `spatial_memory.py` | Async SQLite database for persistent object memory — logs every detection with label, confidence, position, direction, timestamp. Supports search, dedup (30s cooldown), environment context, and session-scoped queries |
| `maps_api.py` (internal) | IP geolocation fallback (Google Geolocation API → ipinfo.io), geocoding (address → coords), Haversine distance calculation, HTML-to-speech instruction cleaning, daily quota guards (stays within $200/month free tier) |
| `providers.py` | Multi-VLM fallback chain (Gemini → Grok → Azure OpenAI → NVIDIA NIM → HuggingFace) with health tracking, cooldown periods, and automatic failover — used by `read_text_in_scene` and `describe_scene_detailed` |

---

| Component | Technology |
|-----------|-----------|
| Core Orchestration | Vision Agents SDK (`vision-agents>=0.3.7`) |
| Reasoning Engine | Gemini 2.5 Flash Realtime (speech-to-speech) |
| Scene Analysis | Multi-VLM fallback (Gemini → Grok → Azure → NVIDIA → HF) |
| Pose Estimation | Ultralytics YOLO11 Pose (`yolo11n-pose.pt`) |
| Object Detection | Ultralytics YOLO11 (`yolo11n.pt`) |
| Hand Tracking | Google MediaPipe Hand Landmarker (21 keypoints/hand) |
| ASL Recognition | MediaPipe finger-state classifier (A, B, D, I, L, V, W, Y, S, 5) |
| NLP Processing | HuggingFace Inference API (gloss → English) |
| OCR / Text Read | OCR Processor (multi-provider VLM) |
| Navigation | Google Maps Directions + Geocoding + Places API |
| Spatial Memory | aiosqlite persistent object memory |
| Transport | GetStream Edge (WebRTC) |
| Tool Calling | Model Context Protocol (MCP) — 12 tools |
| Weather | Open-Meteo API (free, no key required) |
| Frontend | React 19 + Vite 7 + TypeScript 5.9 |
| 3D Avatar | React Three Fiber (built-in fallback) |
| Testing (backend) | pytest + asyncio (24 tests) |
| Testing (frontend) | Vitest + React Testing Library (46 tests) |

---

## Project Structure

```
├── backend/
│   ├── main.py                        # Agent entry point (Gemini Realtime + Stream)
│   ├── providers.py                   # Multi-VLM fallback chain (5 adapters)
│   ├── m5_bridge.py                   # Camera bridge (RTSP/webcam → VideoForwarder)
│   ├── hand_landmarker.task           # MediaPipe hand model (7.5 MB)
│   ├── yolo11n.pt                     # YOLO11 nano detection model
│   ├── yolo11n-pose.pt                # YOLO11 nano pose model
│   ├── processors/
│   │   ├── signbridge_processor.py    # YOLO Pose + MediaPipe → sign translation
│   │   ├── guidelens_processor.py     # YOLO Detection → hazard alerts
│   │   ├── mediapipe_hands.py         # MediaPipe hand landmark module
│   │   └── ocr_processor.py          # OCR + dense scene description
│   ├── mcp_tools/
│   │   ├── maps_api.py               # Google Maps navigation (live API)
│   │   ├── spatial_memory.py          # SQLite spatial object memory
│   │   ├── navigation_engine.py       # Hazard alert queue + route state
│   │   └── smart_tools.py            # Time, weather, emergency, device status
│   ├── utils/
│   │   └── local_storage.py           # Frame store, detection cache, sessions
│   └── tests/
│       ├── test_day4.py               # 14 tests (spatial memory, maps, navigation)
│       └── test_day5.py               # 10 tests (haptics, telemetry, alerts)
├── frontend/
│   ├── src/
│   │   ├── App.tsx                    # Main app — session, polling, layout
│   │   ├── components/
│   │   │   ├── VideoRoom.tsx          # Stream Video SDK WebRTC
│   │   │   ├── StatusBar.tsx          # Connection & mode status
│   │   │   ├── ChatLog.tsx            # Transcript history
│   │   │   ├── TelemetryPanel.tsx     # Enterprise metrics dashboard
│   │   │   ├── AlertOverlay.tsx       # Hazard warnings (audio + visual + haptic)
│   │   │   ├── Avatar3D/             # 3D avatar with lip-sync
│   │   │   ├── OCROverlay.tsx         # Text detection overlay
│   │   │   └── ProviderSelector.tsx   # VLM provider management
│   │   ├── hooks/
│   │   │   └── useAgentSession.ts     # Session lifecycle hook
│   │   ├── utils/
│   │   │   └── api.ts                 # Backend API client
│   │   └── types/
│   │       └── index.ts               # TypeScript interfaces
│   └── package.json
├── testing/                           # Integration test suites
├── Planning/
│   ├── DayWise Plan.md                # Day-by-day implementation log
│   └── Strategic Plan.md
├── Documentation/                     # Vision Agents SDK reference (HTML)
└── Architecture.md
```

---

## Day-by-Day Progress

| Day | Focus | Highlights |
|-----|-------|------------|
| **1** | Infrastructure | Vision Agents SDK, Stream WebRTC, React frontend, dual-mode toggle, camera bridge |
| **2** | Vision Processors | YOLO11 Pose (SignBridge) + YOLO11 Detection (GuideLens), multi-VLM provider chain, mode switching |
| **3** | Advanced Visuals | OCR processor, NVIDIA VLM integration, 3D Avatar with lip-sync, OCR overlay |
| **4** | Agentic Tools | Google Maps live API, SQLite spatial memory, MediaPipe hand landmarks, ASL finger-spelling, navigation engine |
| **5** | Synthesis & Polish | 12 MCP tools (time, weather, emergency, color ID, device status), glassmorphism UI, AlertOverlay v2 (Web Audio + severity), enterprise metrics, 24 tests |

---

## Testing

```bash
# Backend tests (24 tests)
cd backend
uv run python -m pytest tests/ -v

# Day 4 tests only (spatial memory, maps, navigation)
uv run python -m pytest tests/test_day4.py -v

# Day 5 tests only (haptics, telemetry, alerts)
uv run python -m pytest tests/test_day5.py -v

# Frontend tests (46 tests)
cd frontend
npx vitest run --reporter verbose
```

---

## License

MIT
