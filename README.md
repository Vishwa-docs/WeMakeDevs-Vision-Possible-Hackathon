# 🌍 WorldLens

> For **466 million** people with disabling hearing loss and **43 million** with visual impairment, _"What did you say?"_ and _"What's in front of me?"_ are not small questions. They are daily barriers.

**WorldLens** is an autonomous, real-time assistive vision platform that turns a camera feed into an intelligent companion — one that can see, speak, navigate, and translate sign language. Built as a dual-mode system, it powers two distinct personas:

- **👁️ GuideLens** — A walking navigation and environmental awareness assistant for visually impaired users. It sees what's in front of them, reads signs, warns about hazards, gives turn-by-turn directions, and describes the world — all through natural voice conversation.
- **🤟 SignBridge** — A real-time sign language translation bridge using YOLO11 Pose estimation and MediaPipe hand tracking to interpret ASL finger-spelling and gestures into spoken English. _(Prototype stage — see Future Plans)_

> Built for the **WeMakeDevs Vision Possible Hackathon** (February 2026)

> **Note:** Architecture and idea were designed by me. Used AI for code generation (Perplexity + GitHub Copilot) as coding and refinement agents.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [GuideLens — Navigation Mode](#guidelens--navigation-mode)
- [SignBridge — Sign Language Mode](#signbridge--sign-language-mode)
- [MCP Tools (Model Context Protocol)](#mcp-tools-model-context-protocol)
- [M5Stack K210 Camera (Edge Device)](#m5stack-k210-camera-edge-device)
- [Local Setup Guide](#local-setup-guide)
- [Docker Deployment](#docker-deployment)
- [API Keys](#api-keys)
- [Vision Agents SDK Integration](#vision-agents-sdk-integration)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Future Plans](#future-plans)
- [License](#license)

---

## How It Works

1. **Camera** captures live video — from a webcam, phone camera, or an M5Stack K210 edge device
2. **GetStream Edge Network** transports video/audio via WebRTC with sub-200ms latency
3. **Vision Agents Backend** processes every frame through YOLO11 object detection, estimates distances and approach speeds, runs OCR, and feeds everything to Gemini 2.5 Flash Realtime
4. **Gemini 2.5 Flash Realtime** — the reasoning engine — sees the video, hears the user's voice, and speaks back naturally. It autonomously calls MCP tools (Google Maps directions, spatial memory, weather, etc.) when needed
5. **React 19 Frontend** displays the video call, hazard alerts, OCR overlays, navigation status, and a 3D avatar for SignBridge mode

The entire system runs as a single real-time voice+vision conversation — the user speaks, the AI sees and responds, with no buttons to press or screens to read.

> **⚠️ Network connection is required.** WorldLens relies on cloud APIs (GetStream WebRTC, Google Gemini, Google Maps) for real-time processing. An active internet connection is essential for all features.

---

## Architecture

```
Camera (Webcam / M5Stack K210)
    │
    ▼
GetStream Edge Network (WebRTC — real-time video + audio transport)
    │
    ▼
┌─────────────────────── Vision Agents Backend ───────────────────────┐
│                                                                      │
│  ┌──────────────────┐   ┌──────────────────┐   ┌─────────────────┐  │
│  │  YOLO11 Detection │   │  YOLO11 Pose     │   │  OCR Processor  │  │
│  │  (GuideLens)      │   │  + MediaPipe     │   │  (Multi-VLM     │  │
│  │  80 COCO classes  │   │  (SignBridge)    │   │   fallback)     │  │
│  │  Hazard tracking  │   │  17 body + 42   │   │  Gemini → Grok  │  │
│  │  Distance/dir est │   │  hand keypoints  │   │  → Azure → NV  │  │
│  └────────┬─────────┘   └────────┬─────────┘   └───────┬─────────┘  │
│           │                      │                      │            │
│           ▼                      ▼                      ▼            │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │              Event Bus (BaseEvent pub/sub)                     │ │
│  │  HazardDetectedEvent │ SceneSummaryEvent │ OCRResultEvent      │ │
│  │  SignDetectedEvent   │ GestureBufferEvent│ SignTranslationEvent│ │
│  └──────────────────────────────┬──────────────────────────────────┘ │
│                                 │                                    │
│                                 ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────┐│
│  │           Gemini 2.5 Flash Realtime (LLM)                       ││
│  │           Speech-to-Speech @ 5 FPS                               ││
│  │           + 12 MCP Tools (Maps, Memory, Weather, Haptics, ...)  ││
│  └──────────────────────────────────────────────────────────────────┘│
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
    │
    ▼
React 19 Frontend (Stream Video SDK + 3D Avatar + Alert Overlay)
```

> See [Architecture.md](Architecture.md) for the detailed Mermaid data-flow diagram.

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Orchestration** | [Vision Agents SDK](https://pypi.org/project/vision-agents/) `>=0.3.7` | Agent lifecycle, processors, events, MCP |
| **Reasoning LLM** | Gemini 2.5 Flash Realtime | Speech-to-speech voice + vision reasoning |
| **Scene Analysis** | Multi-VLM fallback chain | Gemini → Grok → Azure GPT-4o → NVIDIA Cosmos → HuggingFace |
| **Object Detection** | Ultralytics YOLO11 (`yolo11n.pt`) | 80-class detection, distance/direction estimation |
| **Pose Estimation** | Ultralytics YOLO11 Pose (`yolo11n-pose.pt`) | 17 COCO body keypoints for sign language |
| **Hand Tracking** | Google MediaPipe Hand Landmarker | 21 keypoints per hand, finger-state ASL recognition |
| **NLP** | HuggingFace Inference API | Gloss → English translation (Llama 3) |
| **Navigation** | Google Maps Directions + Geocoding + Places API | Turn-by-turn walking directions |
| **Spatial Memory** | aiosqlite (async SQLite) | Persistent object detection history |
| **Weather** | Open-Meteo API | Free weather data, no API key needed |
| **WebRTC Transport** | GetStream Edge Network | Real-time video/audio with global edge CDN |
| **Frontend** | React 19 + Vite 7 + TypeScript 5.9 | SPA with WebRTC, 3D avatar, alert overlays |
| **3D Avatar** | React Three Fiber + Three.js | Lip-sync avatar for SignBridge mode |
| **Edge Camera** | M5Stack UnitV K210 | RISC-V AI camera with on-device YOLO v2 tiny |
| **Containerization** | Docker (multi-stage build) | Single-container deployment |
| **Backend Runtime** | Python 3.12, uv package manager | Fast dependency resolution |
| **Testing** | pytest (backend, 24 tests) + Vitest (frontend, 46 tests) | Unit and integration tests |

---

## GuideLens — Navigation Mode

GuideLens is the primary mode — a walking companion that acts as the user's eyes.

### What It Does

- **Object Detection** — YOLO11 detects people, vehicles, obstacles, furniture, animals (80 classes) in every frame
- **Hazard Tracking** — Bounding box growth rate estimation detects approaching objects; direction (left/center/right) and distance (near/medium/far) are continuously computed
- **Proactive Voice Commentary** — The agent describes the environment, reads signs, warns about obstacles — all through natural speech via Gemini Realtime
- **Turn-by-Turn Navigation** — User says _"Take me to L&T South City"_ and the agent calls Google Maps, reads the route aloud, and guides step-by-step
- **Text Reading (OCR)** — Signs, building names, bus numbers, notices are read aloud using a multi-VLM provider chain
- **Spatial Memory** — Every detected object is logged to SQLite with timestamp, position, and direction. User can ask _"When did you last see a person?"_
- **Haptic Alerts** — Critical hazards trigger visual + audio + haptic feedback on the frontend
- **Weather-Aware Greeting** — On session start, the agent greets with current time, date, and weather conditions

### GuideLens Vision Pipeline

```
Camera Frame → YOLO11 Detection (80 classes) → BboxTracker
                                                    │
                                        Direction + Distance estimation
                                        Growth rate (approach speed)
                                                    │
                                        ┌───────────┴───────────┐
                                        ▼                       ▼
                                HazardDetectedEvent     SceneSummaryEvent
                                → Haptic Alert          → Voice Description
                                → Urgent warning        → Continuous commentary
```

---

## SignBridge — Sign Language Mode

SignBridge translates sign language to spoken English using computer vision.

> **Status:** Prototype stage. Core detection pipeline works. Full two-user bridge mode (deaf user ↔ hearing user) was planned but deprioritized due to time constraints.

### What It Does

- **Body Pose Tracking** — YOLO11 Pose extracts 17 body keypoints for gesture classification
- **Hand Landmark Tracking** — MediaPipe detects 21 hand keypoints per hand, analyzes finger states (extended/curled)
- **ASL Letter Recognition** — Recognizes fingerspelled letters: A, B, D, I, L, V, W, Y, S, 5
- **Gesture Classification** — 30-frame buffer classifies gestures (wave, point, thumbs up, open palm, etc.)
- **NLP Translation** — HuggingFace Llama 3 converts ASL gloss to natural English

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

---

## MCP Tools (Model Context Protocol)

WorldLens registers **12 MCP tools** via the Vision Agents SDK's `@agent.llm.register_function()`. Gemini autonomously decides when to call each tool based on conversation context — no manual triggers needed.

| # | Tool | What It Does | Example Trigger |
|---|------|-------------|-----------------|
| 1 | `read_text_in_scene` | Extracts text from the camera frame via multi-VLM chain | _"What does that sign say?"_ |
| 2 | `describe_scene_detailed` | Dense VLM scene description | _"What's around me?"_ |
| 3 | `get_walking_directions` | Google Maps turn-by-turn walking directions | _"Take me to the train station"_ |
| 4 | `search_nearby_places` | Finds nearest pharmacy, bus stop, ATM, etc. | _"Where's the nearest hospital?"_ |
| 5 | `search_memory` | Queries SQLite spatial memory for past detections | _"When did you last see a dog?"_ |
| 6 | `get_environment_context` | Combines navigation state + memory history | _"What's been happening around me?"_ |
| 7 | `trigger_haptic_alert` | Sends visual + audio + haptic alert to device | Auto-triggered for approaching vehicles |
| 8 | `get_time_and_date` | Current local time, date, day of week | _"What time is it?"_ |
| 9 | `get_weather` | Weather via Open-Meteo (free, no key) | _"Do I need an umbrella?"_ |
| 10 | `identify_colors` | Describes colors of objects in view | _"What color is that car?"_ |
| 11 | `emergency_alert` | Logs emergency, notifies contacts (production) | _"Help!"_ or danger detected |
| 12 | `get_device_status` | Battery, camera/mic status, system uptime | _"Is the camera working?"_ |

### Supporting Infrastructure

| Module | Purpose |
|--------|---------|
| `navigation_engine.py` | Priority-based hazard announcement queue with cooldowns, route step tracking, environment state machine |
| `spatial_memory.py` | Async SQLite database — logs every detection with label, confidence, position, direction, timestamp |
| `maps_api.py` | IP geolocation fallback, geocoding, Haversine distance, HTML-to-speech instruction cleaning, quota guards |
| `providers.py` | Multi-VLM fallback chain with health tracking, cooldowns, and automatic failover across 5 providers |

---

## M5Stack K210 Camera (Edge Device)

WorldLens supports the **M5Stack UnitV K210** as a standalone edge camera. The K210 is a RISC-V chip with a hardware KPU neural accelerator that runs YOLO v2 tiny on-device at ~15 FPS.

```
M5Stack UnitV K210 (OV2640 sensor)
    │  On-device YOLO v2 tiny (20 classes, ~1.3 MiB model)
    │  640×480 capture → 224×224 KPU inference
    ▼
USB/UART Serial (115200 baud)
    │  JPEG frames + JSON detections
    ▼
Host Bridge (Python, port 8001)
    │  WebSocket server + Web UI
    ▼
Browser Dashboard (http://localhost:8001)
    │  Live feed + bounding boxes + detection log
```

- **20 detection classes:** person, car, bus, bicycle, motorbike, chair, bottle, dog, cat, and more
- **Hardware:** 8 MiB SRAM, 16 MB Flash, 400 MHz dual-core RISC-V, 0.8 TOPS KPU
- **Firmware:** MaixPy (MicroPython for K210)

> See [m5stack_camera/README.md](m5stack_camera/README.md) for flashing instructions and setup guide.

---

## Local Setup Guide

### Prerequisites

- **Python 3.12+** with [uv](https://docs.astral.sh/uv/) package manager
- **Node.js 18+** (recommended: 20 LTS)
- **Webcam** or phone camera (or M5Stack K210)
- **Active internet connection** — required for all cloud APIs
- API keys (see [API Keys](#api-keys) section)

### 1. Clone the Repository

```bash
git clone https://github.com/Vishwa-docs/WeMakeDevs-Vision-Possible-Hackathon.git
cd WeMakeDevs-Vision-Possible-Hackathon
```

### 2. Backend Setup

```bash
cd backend

# Install all dependencies (creates .venv automatically)
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your API keys (see API Keys section below)

# Run as HTTP server (for frontend connection)
uv run main.py serve --host 0.0.0.0 --port 8000
```

The backend will start on `http://localhost:8000`. You should see:
```
Stream agent user upserted: worldlens-agent
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure environment
cp .env.example .env
# Set VITE_STREAM_API_KEY (same as backend's STREAM_API_KEY)
# Set VITE_BACKEND_URL=http://localhost:8000

# Start dev server
npm run dev
```

The frontend will start at `http://localhost:5173`. Open it in your browser to begin a session.

### 4. Start a Session

1. Open `http://localhost:5173` in your browser
2. Allow camera and microphone access
3. The agent will greet you with current time and weather
4. Say _"Take me to [destination]"_ to start navigation, or just talk naturally

> **Tip:** The backend defaults to GuideLens mode. Use the mode toggle in the UI or `POST /switch-mode` to switch to SignBridge.

---

## Docker Deployment

WorldLens can be deployed as a single Docker container that includes both the backend and a pre-built frontend.

```bash
# Build the image (use linux/amd64 for mediapipe compatibility)
docker build --platform linux/amd64 -f deploy/Dockerfile -t worldlens:latest .

# Run with your .env file
docker run --platform linux/amd64 -p 8000:8000 --env-file .env worldlens:latest
```

The container serves:
- **Backend API** at `http://localhost:8000/`
- **Frontend** at `http://localhost:8000/` (static files served by FastAPI)

> See [deploy/README.md](deploy/README.md) for detailed Docker setup, docker-compose configuration, and environment variable reference.

---

## API Keys

### Required (App will not function without these)

| Variable | Service | How to Get |
|----------|---------|------------|
| `STREAM_API_KEY` | GetStream WebRTC | [getstream.io/dashboard](https://getstream.io/dashboard/) — create an app, copy API Key |
| `STREAM_API_SECRET` | GetStream WebRTC | Same dashboard — copy the Secret |
| `GOOGLE_API_KEY` | Gemini 2.5 Flash Realtime | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — click "Create API Key" |

### Recommended (Enhanced navigation features)

| Variable | Service | How to Get | Used For |
|----------|---------|------------|----------|
| `MAPS_API_KEY` | Google Maps Platform | [console.cloud.google.com](https://console.cloud.google.com/apis/credentials) — enable Directions, Geocoding, Places, Geolocation APIs | Walking directions, nearby places, location info |
| `HF_API_TOKEN` | HuggingFace | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | SignBridge NLP (gloss → English via Llama 3) |

### Optional (VLM fallback providers — automatic failover)

| Variable | Service | Used For |
|----------|---------|----------|
| `NGC_API_KEY` | NVIDIA NGC | Cosmos 2 VLM for dense scene descriptions |
| `XAI_API_KEY` | xAI Grok | Grok Vision for OCR and scene analysis |
| `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_KEY` / `AZURE_OPENAI_DEPLOYMENT` | Azure OpenAI | GPT-4o Vision fallback |

### Google Maps — Detailed Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select existing)
3. Navigate to **APIs & Services → Library** and enable:
   - **Directions API** — walking turn-by-turn navigation
   - **Geocoding API** — address → coordinates
   - **Places API (Text Search)** — "nearest pharmacy" queries
   - **Geolocation API** — IP-based approximate location
4. Go to **APIs & Services → Credentials** → Create API Key
5. (Recommended) Restrict the key to only the 4 APIs above
6. Set `MAPS_API_KEY` in your `.env`

> **Without `MAPS_API_KEY`**, the navigation tools still work but return mock directions. The app is fully functional for demo purposes without this key.

---

## Vision Agents SDK Integration

WorldLens is built on top of the **Vision Agents SDK** (`vision-agents>=0.3.7`). Every core capability uses the SDK's classes and APIs.

| SDK Class | Where Used | Purpose |
|-----------|-----------|---------|
| `Agent` | `main.py` → `create_agent()` | Core agent instance — LLM, processors, event bus, conversation |
| `AgentLauncher` | `main.py` → entry point | Manages agent lifecycle: creation, call joining, concurrency, timeouts |
| `Runner` | `main.py` → entry point | Top-level entry — launches as FastAPI server (`serve`) or standalone (`run`) |
| `gemini.Realtime` | `create_agent()` | LLM backend — Gemini 2.5 Flash Realtime at 5 FPS |
| `getstream.Edge` | `create_agent()` | WebRTC transport — GetStream Edge Network |
| `VideoProcessorPublisher` | All 3 processors | Base class for video processors that publish annotated frames |
| `BaseEvent` | All custom events | Base class for `HazardDetectedEvent`, `SignDetectedEvent`, etc. |
| `agent.llm.register_function` | 12 MCP tools | Registers tools that Gemini can call autonomously |
| `agent.simple_response()` | Event handlers | Sends text prompts to LLM for immediate spoken response |

### How the SDK Powers Each Feature

1. **Real-time Voice** — `gemini.Realtime(fps=5)` provides full speech-to-speech reasoning over live video
2. **WebRTC Transport** — `getstream.Edge()` manages the WebRTC connection with global edge CDN
3. **Video Processors** — SignBridge, GuideLens, and OCR processors extend `VideoProcessorPublisher` / `VideoProcessor`
4. **Event System** — Custom events (`HazardDetectedEvent`, `SceneSummaryEvent`, etc.) use `BaseEvent` pub/sub
5. **MCP Tool Calling** — 12 tools via `@agent.llm.register_function()` — Gemini decides when to call each
6. **Agent Lifecycle** — `Runner` → `AgentLauncher` → `create_agent()` → `join_call()` → `agent.finish()`

---

## Project Structure

```
WorldLens/
├── backend/                           # Python backend (Vision Agents SDK)
│   ├── main.py                        # Agent entry point — Gemini Realtime + Stream + 12 MCP tools
│   ├── providers.py                   # Multi-VLM fallback chain (5 providers)
│   ├── m5_bridge.py                   # Camera bridge (RTSP/webcam → VideoForwarder)
│   ├── hand_landmarker.task           # MediaPipe hand model (7.5 MB)
│   ├── yolo11n.pt                     # YOLO11 nano detection model
│   ├── yolo11n-pose.pt                # YOLO11 nano pose model
│   ├── processors/                    # Video processing pipelines
│   │   ├── guidelens_processor.py     # YOLO11 Detection → hazard alerts
│   │   ├── signbridge_processor.py    # YOLO11 Pose + MediaPipe → sign translation
│   │   ├── ocr_processor.py           # Multi-VLM OCR + dense scene description
│   │   └── mediapipe_hands.py         # MediaPipe 21-keypoint hand landmarker
│   ├── mcp_tools/                     # MCP tool implementations
│   │   ├── maps_api.py                # Google Maps navigation (Directions + Places)
│   │   ├── spatial_memory.py          # SQLite spatial object memory
│   │   ├── navigation_engine.py       # Hazard announcement queue + route tracking
│   │   └── smart_tools.py            # Time, weather, emergency, color ID, device status
│   ├── utils/
│   │   └── local_storage.py           # Frame store, detection cache, sessions
│   └── tests/
│       ├── test_day4.py               # 14 tests (spatial memory, maps, navigation)
│       └── test_day5.py               # 10 tests (haptic alerts, telemetry)
│
├── frontend/                          # React 19 + TypeScript + Vite 7
│   ├── src/
│   │   ├── App.tsx                    # Main app — session, polling, layout
│   │   ├── components/
│   │   │   ├── VideoRoom.tsx          # Stream Video SDK WebRTC call
│   │   │   ├── StatusBar.tsx          # Connection + mode status
│   │   │   ├── ChatLog.tsx            # Transcript history
│   │   │   ├── AlertOverlay.tsx       # Hazard alerts (audio + visual + haptic)
│   │   │   ├── TelemetryPanel.tsx     # Real-time metrics dashboard
│   │   │   ├── OCROverlay.tsx         # Text detection overlay
│   │   │   ├── Avatar3D/             # 3D avatar with lip-sync
│   │   │   └── ProviderSelector.tsx   # VLM provider management
│   │   ├── hooks/useAgentSession.ts   # Session lifecycle hook
│   │   ├── utils/api.ts              # Backend API client
│   │   └── types/index.ts            # TypeScript interfaces
│   └── package.json
│
├── deploy/                            # Docker deployment
│   ├── Dockerfile                     # Multi-stage build (Node + Python)
│   ├── docker-compose.yml             # Single-command deployment
│   ├── .env.docker                    # Template env file
│   └── build.sh                       # Build helper script
│
├── m5stack_camera/                    # M5Stack K210 edge camera
│   ├── camera/                        # K210 MicroPython firmware code
│   ├── camera_host/                   # Host bridge + Web UI
│   └── models/                        # YOLO v2 tiny model for K210
│
├── testing/                           # Integration test suites
├── Plans/                             # Day-by-day planning docs
├── Documentation/                     # Vision Agents SDK reference (HTML)
├── Architecture.md                    # Mermaid architecture diagram
└── README.md                          # ← You are here
```

---

## Testing

### Backend (24 tests)

```bash
cd backend

# Run all tests
uv run python -m pytest tests/ -v

# Day 4 tests (spatial memory, maps, navigation engine)
uv run python -m pytest tests/test_day4.py -v

# Day 5 tests (haptic alerts, telemetry, emergency)
uv run python -m pytest tests/test_day5.py -v
```

### Frontend (46 tests)

```bash
cd frontend

# Run all tests
npx vitest run --reporter verbose
```

---

## Future Plans

- **Full Edge Deployment with SIM Card** — Run WorldLens on a mobile device with SIM card connectivity for fully portable, untethered navigation assistance. The M5Stack K210 paired with a 4G-enabled SBC (Raspberry Pi + SIM HAT) would enable outdoor use without WiFi dependency
- **Lip Reading to Speech** — Computer vision-based lip reading to supplement audio input in noisy environments
- **Caller Alerts via Vibration** — Detect when someone is speaking to/calling out to the user and alert via haptic vibration patterns
- **Full SignBridge Two-User Mode** — Complete the bridge between a sign language user and a hearing user, with real-time bidirectional translation
- **Expanded ASL Recognition** — Full ASL vocabulary beyond finger-spelling, including common phrases and conversational signs
- **Multi-language Support** — Extend GuideLens navigation and voice to additional languages
- **Offline Fallback** — On-device YOLO inference + edge TTS for basic hazard detection without internet

---

## Day-by-Day Build Progress

| Day | Focus | Key Deliverables |
|-----|-------|-----------------|
| **1** | Infrastructure | Vision Agents SDK integration, GetStream WebRTC, React frontend skeleton, dual-mode toggle, camera bridge |
| **2** | Vision Processors | YOLO11 Pose (SignBridge) + YOLO11 Detection (GuideLens), multi-VLM provider chain, mode switching |
| **3** | Advanced Visuals | OCR processor, NVIDIA VLM integration, 3D Avatar with lip-sync, OCR overlay |
| **4** | Agentic Tools | Google Maps live API, SQLite spatial memory, MediaPipe hand landmarks, ASL finger-spelling, navigation engine |
| **5** | Synthesis & Polish | 12 MCP tools, AlertOverlay v2 (Web Audio + severity), enterprise telemetry, glassmorphism UI, 24 backend + 46 frontend tests |

---

## License

MIT
