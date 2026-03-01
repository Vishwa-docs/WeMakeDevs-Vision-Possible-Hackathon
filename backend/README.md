# WorldLens Backend

The Python backend powering WorldLens — built on the **Vision Agents SDK** with **Gemini 2.5 Flash Realtime** for speech-to-speech reasoning, **YOLO11** for real-time object/pose detection, and **12 MCP tools** for autonomous navigation, memory, and environmental awareness.

---

## Prerequisites

- **Python 3.12+** — required by the Vision Agents SDK
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager (recommended) or pip
- **Webcam** — or M5Stack K210 camera, or phone camera via WebRTC
- **Active internet connection** — required for GetStream, Gemini, and Google Maps APIs

## Quick Start

```bash
# 1. Install all dependencies (creates .venv automatically)
uv sync

# 2. Configure environment
cp .env.example .env
# Edit .env and add your API keys (see API Keys section below)

# 3. Run as HTTP server (for frontend connection)
uv run main.py serve --host 0.0.0.0 --port 8000

# 4. Or run in standalone CLI mode (no frontend needed)
uv run main.py run
```

The server starts at `http://localhost:8000`. You should see:
```
Stream agent user upserted: worldlens-agent
```

---

## Architecture

```
main.py                          # Entry point — Agent + AgentLauncher + Runner
                                 # Creates Gemini 2.5 Flash Realtime LLM
                                 # Registers 12 MCP tools
                                 # Subscribes to 13 event handlers
                                 # 20+ FastAPI REST endpoints
                                 # Programmatic greeting (time + weather)
│
├── providers.py                 # Multi-VLM fallback chain (5 providers)
│   Azure GPT-4o → Gemini Pro → xAI Grok → NVIDIA Cosmos → HuggingFace
│
├── m5_bridge.py                 # Camera bridge (RTSP/webcam → VideoForwarder)
│
├── processors/
│   ├── guidelens_processor.py   # YOLO11 Detection → hazard tracking → events
│   │   └── BboxTracker          # Approach speed via growth rate estimation
│   │   └── Direction/distance   # left/center/right, near/medium/far
│   │
│   ├── signbridge_processor.py  # YOLO11 Pose (17 keypoints) → gesture buffer
│   │   └── GestureClassifier    # 30-frame window → wave/point/thumbs up/etc.
│   │   └── GlossTranslator     # HuggingFace NLP or rule-based fallback
│   │
│   ├── ocr_processor.py         # Passive frame capture → multi-VLM OCR
│   │   └── read_text()          # Extract visible text from camera
│   │   └── describe_scene()     # Dense VLM scene description
│   │
│   └── mediapipe_hands.py       # MediaPipe 21-keypoint hand landmarker
│       └── Finger state analysis → ASL letter recognition (A,B,D,I,L,V,W,Y,S,5)
│
├── mcp_tools/
│   ├── maps_api.py              # Google Maps Directions + Places + Geocoding
│   │   └── IP geolocation fallback (Google Geolocation → ipinfo.io)
│   │   └── Haversine distance, HTML→speech cleaning, quota guards
│   │
│   ├── spatial_memory.py        # Async SQLite object memory
│   │   └── Logs every detection: label, confidence, position, direction, timestamp
│   │   └── Search, dedup (30s cooldown), environment context queries
│   │
│   ├── navigation_engine.py     # Smart announcement system
│   │   └── SmartAnnouncer: priority queue with per-level cooldowns
│   │   └── Route step tracking, environment state machine
│   │   └── User speech suppression to prevent talking over user
│   │
│   └── smart_tools.py           # Utility tools
│       └── get_time_and_date, get_weather (Open-Meteo, free)
│       └── emergency_alert, identify_colors, get_device_status
│
├── utils/
│   └── local_storage.py         # Frame snapshots, detection cache, session mgmt

```

---

## Models

| Model | File | Size | Purpose |
|-------|------|------|---------|
| YOLO11 Nano Detection | `yolo11n.pt` | ~6.5 MB | 80 COCO-class object detection (GuideLens) |
| YOLO11 Nano Pose | `yolo11n-pose.pt` | ~6.5 MB | 17 body keypoint estimation (SignBridge) |
| MediaPipe Hand Landmarker | `hand_landmarker.task` | ~7.5 MB | 21 hand keypoints per hand, finger state |
| Gemini 2.5 Flash Realtime | Cloud API | — | Speech-to-speech LLM reasoning @ 5 FPS |

Both YOLO11 models are from [Ultralytics](https://docs.ultralytics.com/models/yolo11/) and run locally via the `ultralytics` Python package. MediaPipe runs locally via Google's `mediapipe` package. Gemini is accessed via the Vision Agents SDK's `gemini.Realtime` plugin.

---

## MCP Tools (12 Total)

All tools are registered via `@agent.llm.register_function()` — the Vision Agents SDK's implementation of the **Model Context Protocol**. Gemini decides autonomously when to call each tool.

| # | Tool Name | Source Module | Description |
|---|-----------|--------------|-------------|
| 1 | `read_text_in_scene` | `ocr_processor.py` | OCR text extraction from current camera frame via multi-VLM chain |
| 2 | `describe_scene_detailed` | `ocr_processor.py` | Dense scene description via VLM |
| 3 | `get_walking_directions` | `maps_api.py` | Google Maps turn-by-turn walking directions |
| 4 | `search_nearby_places` | `maps_api.py` | Find nearby POIs (pharmacy, bus stop, ATM, hospital) |
| 5 | `search_memory` | `spatial_memory.py` | Query SQLite spatial memory for past detections |
| 6 | `get_environment_context` | `spatial_memory.py` + `navigation_engine.py` | Combined nav state + memory history |
| 7 | `trigger_haptic_alert` | Built-in (`main.py`) | Visual + audio + haptic alert with severity and direction |
| 8 | `get_time_and_date` | `smart_tools.py` | Current local time, date, day of week |
| 9 | `get_weather` | `smart_tools.py` | Weather via Open-Meteo API (free, no key required) |
| 10 | `identify_colors` | `smart_tools.py` | Color identification of objects in view |
| 11 | `emergency_alert` | `smart_tools.py` | Emergency logging + contact notification (production) |
| 12 | `get_device_status` | `smart_tools.py` | Battery, camera/mic status, system uptime |

---

## VLM Provider Fallback Chain

The multi-provider chain in `providers.py` provides resilient OCR and scene description. Each provider has lazy initialization, health checks, and cooldowns:

```
Azure GPT-4o → Gemini Pro Vision → xAI Grok → NVIDIA Cosmos 2 → HuggingFace
  (primary)       (fast)            (fast)      (dense desc)     (free tier)
```

If one provider fails, the chain automatically tries the next. Configure via environment variables:

| Variable | Provider |
|----------|---------|
| `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_KEY` + `AZURE_OPENAI_DEPLOYMENT` | Azure GPT-4o Vision |
| `GOOGLE_API_KEY` | Gemini Pro Vision |
| `XAI_API_KEY` | xAI Grok Vision |
| `NGC_API_KEY` | NVIDIA Cosmos 2 |
| `HF_API_TOKEN` | HuggingFace (free tier) |

---

## Event System

Custom events extend the SDK's `BaseEvent` and flow through the event bus:

| Event | Emitted By | Purpose |
|-------|-----------|---------|
| `ObjectDetectedEvent` | `guidelens_processor.py` | Object list per frame |
| `HazardDetectedEvent` | `guidelens_processor.py` | Approaching obstacle with direction, distance, growth rate |
| `SceneSummaryEvent` | `guidelens_processor.py` | Periodic scene summary for proactive commentary |
| `OCRResultEvent` | `ocr_processor.py` | Extracted text from camera frame |
| `SceneDescriptionEvent` | `ocr_processor.py` | VLM scene description |
| `SignDetectedEvent` | `signbridge_processor.py` | Person + hand count, ASL letters |
| `GestureBufferEvent` | `signbridge_processor.py` | Classified gesture from 30-frame window |
| `SignTranslationEvent` | `signbridge_processor.py` | Translated sign language gloss → English |

---

## API Endpoints

### Core

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/mode` | Current agent mode (`guidelens` / `signbridge`) |
| `POST` | `/switch-mode` | Toggle between modes |
| `POST` | `/set-mode/{mode}` | Set specific mode |
| `GET` | `/token?user_id=` | Generate Stream user token |
| `GET` | `/stream-config` | Stream API key for frontend |
| `GET` | `/transcript` | Conversation transcript |
| `DELETE` | `/transcript` | Clear transcript |

### VLM Providers

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/providers` | Status of all VLM providers |
| `POST` | `/providers/preferred/{id}` | Set preferred provider |
| `GET` | `/providers/fallback-events` | Fallback event log |

### OCR / Scene Description

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/ocr-results` | Cached OCR text results |
| `POST` | `/ocr/read` | Trigger OCR on current frame |
| `POST` | `/ocr/describe` | Trigger dense scene description |

### Navigation & Telemetry

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/telemetry` | Real-time aggregated metrics |
| `GET` | `/navigation/hazards` | Active hazard alerts |
| `GET` | `/navigation/hazards/poll` | Consumable hazard alerts (cleared after read) |
| `GET` | `/navigation/environment` | Environment summary |

### Smart Tools

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/time` | Current time and date |
| `GET` | `/weather?location=` | Weather for a location |
| `GET` | `/device-status` | Device status info |
| `GET` | `/emergencies` | Emergency alert log |
| `POST` | `/emergency` | Manually trigger emergency |

### Spatial Memory

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/memory/search` | Search spatial memory |
| `GET` | `/memory/recent` | Recent detections |
| `GET` | `/memory/summary` | Memory statistics |

---

## API Keys Setup

### Required Keys

| Service | Variable | How to Get |
|---------|----------|------------|
| **GetStream** | `STREAM_API_KEY` + `STREAM_API_SECRET` | [getstream.io/dashboard](https://getstream.io/dashboard/) → Create app → Copy Key & Secret |
| **Google Gemini** | `GOOGLE_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) → Create API Key |

### Recommended Keys

| Service | Variable | How to Get | Used For |
|---------|----------|------------|----------|
| **Google Maps** | `MAPS_API_KEY` | [console.cloud.google.com](https://console.cloud.google.com/apis/credentials) → Enable Directions, Geocoding, Places, Geolocation APIs → Create Key | Walking directions, nearby places, location |
| **HuggingFace** | `HF_API_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | SignBridge NLP (gloss → English) |

### Optional Keys

| Service | Variable | Used For |
|---------|----------|----------|
| **NVIDIA NGC** | `NGC_API_KEY` | Cosmos 2 VLM scene descriptions |
| **xAI Grok** | `XAI_API_KEY` | Grok Vision OCR fallback |
| **Azure OpenAI** | `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_KEY` + `AZURE_OPENAI_DEPLOYMENT` | GPT-4o Vision fallback |
| **Langfuse** | `LANGFUSE_SECRET_KEY` + `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_HOST` | LLM tracing & analytics |

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

> Without `MAPS_API_KEY`, navigation tools return mock directions. The app is fully functional for demo purposes.

---

## Running Tests

```bash
# All tests (24 total)
uv run python -m pytest tests/ -v

# Day 4 tests — spatial memory, maps API, navigation engine
uv run python -m pytest tests/test_day4.py -v

# Day 5 tests — haptic alerts, telemetry, emergency
uv run python -m pytest tests/test_day5.py -v
```

---

## Dependencies

Key packages (full list in `pyproject.toml`):

| Package | Version | Purpose |
|---------|---------|---------|
| `vision-agents` | `>=0.3.7` | Core SDK — agent, processors, events, MCP, WebRTC |
| `ultralytics` | (via vision-agents) | YOLO11 object detection and pose estimation |
| `mediapipe` | `>=0.10.32` | Hand landmark detection for sign language |
| `opencv-python-headless` | `>=4.13.0` | Frame processing, drawing overlays |
| `aiosqlite` | `>=0.22.1` | Async SQLite for spatial memory |
| `httpx` | `>=0.28.1` | Async HTTP client for API calls |
| `python-dotenv` | `>=1.2.1` | Environment variable loading |
| `structlog` | `>=23.3` | Structured logging |
