# WorldLens Backend

Python backend for the WorldLens dual-mode assistive vision platform, built on the **Vision Agents SDK**.

## Setup

```bash
# Create virtual environment and install dependencies (requires uv)
uv sync

# Or with pip
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Fill in your API keys

# Run as server (for frontend)
python main.py serve --host 0.0.0.0 --port 8000

# Run in development mode (CLI only)
python main.py run
```

## Architecture

```
main.py                    # Entry point — Agent, MCP tools, API endpoints
providers.py               # Multi-VLM fallback provider chain (5 adapters)
m5_bridge.py               # Camera bridge (RTSP/webcam → VideoForwarder)
processors/
├── signbridge_processor.py  # YOLO11 pose estimation → sign language translation
├── guidelens_processor.py   # Moondream cloud detection → hazard/object detection
└── ocr_processor.py         # On-demand OCR + dense scene description (Day 3)
mcp_tools/
├── maps_api.py              # Google Maps navigation directions
└── spatial_memory.py        # SQLite-backed object memory database
utils/
└── local_storage.py         # Frame store, detection cache, session management
```

## API Endpoints

### Core
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (SDK built-in) |
| `GET` | `/mode` | Current agent mode (`guidelens` / `signbridge`) |
| `POST` | `/switch-mode` | Toggle between modes |
| `POST` | `/set-mode/{mode}` | Set a specific mode |
| `GET` | `/token?user_id=` | Generate Stream user token |
| `GET` | `/stream-config` | Stream API key for frontend |
| `GET` | `/transcript` | Conversation transcript log |
| `DELETE` | `/transcript` | Clear transcript |

### Providers (Day 2)
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/providers` | Status of all VLM providers |
| `POST` | `/providers/preferred/{id}` | Set preferred VLM provider |
| `GET` | `/providers/fallback-events` | Fallback event log |

### OCR / VLM (Day 3)
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/ocr-results` | Cached OCR text results |
| `POST` | `/ocr/read` | Trigger OCR on current video frame |
| `POST` | `/ocr/describe` | Trigger dense scene description |

## VLM Provider Chain

The backend supports 5 VLM providers with automatic failover:

1. **Gemini** (primary) — Google Gemini Pro Vision
2. **Grok** — xAI Grok Vision
3. **Azure OpenAI** — GPT-4o Vision
4. **NVIDIA** — NGC Cosmos 2
5. **HuggingFace** — Inference API

Providers use lazy environment variable loading and include cooldown/health-check logic. Configure via `.env`.

## MCP Tools

Registered tools available to the Gemini reasoning engine:

- `read_text_in_scene` — OCR text extraction from video frames
- `describe_scene_detailed` — Dense scene description via VLM
- `get_walking_directions` — Google Maps navigation (Day 4)
- `remember_object` / `recall_objects` — Spatial memory (Day 4)

## Environment Variables

See [.env.example](.env.example) for all variables. Key requirements:

| Variable | Required | Purpose |
|----------|----------|---------|
| `STREAM_API_KEY` | ✅ | GetStream video transport |
| `STREAM_API_SECRET` | ✅ | GetStream server auth |
| `GOOGLE_API_KEY` | ✅ | Gemini 2.5 Flash Realtime |
| `NGC_API_KEY` | Day 3 | NVIDIA Cosmos 2 VLM |
| `HF_API_TOKEN` | Day 2 | HuggingFace NLP |
| `XAI_API_KEY` | Optional | Grok VLM fallback |
| `AZURE_OPENAI_*` | Optional | Azure OpenAI fallback |
