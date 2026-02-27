# 🌍 WorldLens

**An autonomous, dual-mode assistive vision platform for accessibility.**

> Built for the WeMakeDevs Vision Possible Hackathon (Feb 2026)

WorldLens leverages the **Vision Agents SDK**, **Gemini 2.5 Flash Realtime**, **GetStream Edge Network**, **Google MediaPipe**, and **multi-provider VLM fallback** to provide two modes of real-time assistance:

- **🤟 SignBridge** — Real-time sign language translation via YOLO11 Pose + MediaPipe Hand Landmarks + HuggingFace NLP
- **👁️ GuideLens** — Environmental awareness for visually impaired users via YOLO11 Detection + MCP tool calling (Maps, Spatial Memory)

> NOTE: Used AI for code generation (ChatGPT / GitHub Copilot) to scaffold the initial architecture, then iteratively refined and optimized based on testing and feedback.

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
                                    │                      │                     │
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

## Tech Stack

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
| 3D Avatar | React Three Fiber + Ready Player Me |
| Testing | pytest + asyncio (24 tests) |

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
cd backend

# Run all 24 tests
uv run python -m pytest tests/ -v

# Day 4 tests only (spatial memory, maps, navigation)
uv run python -m pytest tests/test_day4.py -v

# Day 5 tests only (haptics, telemetry, alerts)
uv run python -m pytest tests/test_day5.py -v
```

---

## License

MIT
