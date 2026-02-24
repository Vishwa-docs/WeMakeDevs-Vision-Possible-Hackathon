# 🌍 WorldLens

**An autonomous, dual-mode assistive vision platform for accessibility.**

> Built for the WeMakeDevs Vision Possible Hackathon (Feb 2026)

WorldLens leverages the **Vision Agents SDK**, **Gemini 2.5 Flash Realtime**, **GetStream Edge Network**, and **NVIDIA Cosmos 2** to provide two modes of real-time assistance:

- **🤟 SignBridge** — Real-time sign language translation via YOLO pose estimation + HuggingFace NLP
- **👁️ GuideLens** — Environmental awareness for visually impaired users via Moondream VLM + MCP tool calling

## Architecture

```
M5StickC Camera → Python Relay → Stream Edge (WebRTC) → Vision Agents Backend
                                                              │
                                    ┌─────────────────────────┼───────────────────┐
                                    │                         │                   │
                              YOLO Pose              Moondream Detection    Gemini Realtime
                              (SignBridge)           (GuideLens)            (Voice + Vision)
                                    │                         │                   │
                              HuggingFace NLP         NVIDIA Cosmos 2       MCP Tools
                              (Grammar Fix)           (Dense Scene)         (Maps/Memory)
                                    │                         │                   │
                                    └─────────────────────────┼───────────────────┘
                                                              │
                                                     React 18 Frontend
                                                  (Stream Video SDK + 3D Avatar)
```

## Quick Start

### Backend

```bash
cd backend

# Install dependencies (requires uv)
uv sync

# Configure environment
cp .env.example .env
# Fill in your API keys in .env

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
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `STREAM_API_KEY` | ✅ | GetStream API key |
| `STREAM_API_SECRET` | ✅ | GetStream API secret |
| `GOOGLE_API_KEY` | ✅ | Gemini 2.5 Flash API key |
| `NVIDIA_API_KEY` | Day 3 | NVIDIA Cosmos 2 VLM |
| `HF_TOKEN` | Day 2 | HuggingFace Inference API |
| `MAPS_API_KEY` | Day 4 | Google Maps Directions |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Core Orchestration | Vision Agents SDK (`vision-agents`) |
| Reasoning Engine | Gemini 2.5 Flash Realtime |
| Scene Analysis | NVIDIA Cosmos 2 VLM |
| NLP Processing | HuggingFace Inference API |
| Pose Estimation | Ultralytics YOLOv11 Pose |
| Object Detection | Moondream Cloud Detection |
| Transport | GetStream Edge (WebRTC) |
| Tool Calling | Model Context Protocol (MCP) |
| Frontend | React 18 + Vite + TypeScript |
| 3D Avatar | React Three Fiber + Ready Player Me |

## Project Structure

```
├── backend/
│   ├── main.py              # Agent entry point (Gemini Realtime + Stream)
│   ├── m5_bridge.py          # Camera bridge (RTSP/webcam → VideoForwarder)
│   ├── processors/
│   │   ├── signbridge_processor.py   # YOLO pose → sign translation
│   │   └── guidelens_processor.py    # Moondream → hazard detection
│   ├── mcp_tools/
│   │   ├── maps_api.py       # Google Maps navigation
│   │   └── spatial_memory.py # Object memory database
│   ├── utils/
│   │   └── local_storage.py  # Frame store, detection cache, sessions
│   ├── requirements.txt
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── App.tsx           # Main app with session management
│   │   ├── components/
│   │   │   ├── VideoRoom.tsx      # Stream Video SDK integration
│   │   │   ├── StatusBar.tsx      # Connection & mode status
│   │   │   ├── ChatLog.tsx        # Transcript history
│   │   │   ├── TelemetryPanel.tsx # Metrics display
│   │   │   └── AlertOverlay.tsx   # Hazard warning overlay
│   │   ├── hooks/
│   │   │   └── useAgentSession.ts # Session management hook
│   │   ├── utils/
│   │   │   └── api.ts            # Backend API client
│   │   └── types/
│   │       └── index.ts          # TypeScript interfaces
│   └── package.json
├── Planning/
│   ├── Strategic Plan.md
│   └── DayWise Plan.md
└── Architecture.md
```

## License

MIT
