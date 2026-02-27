# WorldLens Backend

Python backend for the WorldLens dual-mode assistive vision platform, built on the **Vision Agents SDK**.

## Prerequisites

- **Python 3.12+** — required by the Vision Agents SDK
- **uv** (recommended) or pip — for dependency management
- **Webcam** — or M5StickC Plus RTSP camera

## Quick Start

```bash
# 1. Create virtual environment and install all dependencies
uv sync
# OR with pip:
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 2. Copy and configure environment
cp .env.example .env
# Edit .env and add your API keys (see "API Keys Setup" below)

# 3. Run as server (for frontend connection)
uv run main.py serve --host 0.0.0.0 --port 8000

# 4. Or run in CLI-only development mode
uv run main.py run
```

## API Keys Setup

### Required Keys (App won't work without these)

| Service | Variable | How to Get |
|---------|----------|------------|
| **GetStream** | `STREAM_API_KEY` + `STREAM_API_SECRET` | 1. Create account at [getstream.io](https://getstream.io/dashboard/) → 2. Create an app → 3. Copy API Key & Secret from Dashboard |
| **Google Gemini** | `GOOGLE_API_KEY` | 1. Go to [AI Studio](https://aistudio.google.com/apikey) → 2. Click "Create API Key" → 3. Copy the key |

### Recommended Keys (Enhanced features)

| Service | Variable | How to Get | Used For |
|---------|----------|------------|----------|
| **Google Maps Platform** | `MAPS_API_KEY` | 1. Go to [Google Cloud Console](https://console.cloud.google.com/apis/credentials) → 2. Create a project → 3. Enable **Directions API**, **Geocoding API**, **Places API**, **Geolocation API** → 4. Create an API Key → 5. Copy it | Walking directions, nearby places search, location info. Without this key, navigation uses mock/stub data. |
| **HuggingFace** | `HF_API_TOKEN` | 1. Sign up at [huggingface.co](https://huggingface.co/join) → 2. Go to [Settings → Tokens](https://huggingface.co/settings/tokens) → 3. Create a read token | NLP post-processing for SignBridge (gloss → fluent English via Llama 3) |

### Optional Keys (VLM fallback providers — automatic failover)

| Service | Variable | How to Get | Used For |
|---------|----------|------------|----------|
| **NVIDIA NGC** | `NGC_API_KEY` | [NGC Setup](https://org.ngc.nvidia.com/setup/api-key) | Cosmos 2 VLM for dense scene descriptions |
| **xAI Grok** | `XAI_API_KEY` | [console.x.ai](https://console.x.ai/) | Grok Vision fallback |
| **Azure OpenAI** | `AZURE_OPENAI_*` | [Azure Portal](https://portal.azure.com/) → Azure OpenAI resource | GPT-4o Vision fallback |
| **Langfuse** | `LANGFUSE_*` | [cloud.langfuse.com](https://cloud.langfuse.com/) | LLM tracing & analytics |

### Google Maps API — Detailed Setup

The Google Maps integration requires 4 APIs enabled on your GCP project:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select existing)
3. Navigate to **APIs & Services → Library**
4. Search for and enable each:
   - **Directions API** — walking turn-by-turn navigation
   - **Geocoding API** — address → coordinates
   - **Places API (Text Search)** — "nearest pharmacy" queries
   - **Geolocation API** — IP-based approximate location
5. Navigate to **APIs & Services → Credentials**
6. Click **Create Credentials → API Key**
7. (Recommended) Restrict the key to only the 4 APIs above
8. Copy the key to `MAPS_API_KEY` in your `.env`

> **Note:** Without `MAPS_API_KEY`, the navigation tools still work but return mock/stub directions. The app is fully functional for demo purposes without this key.

### Ready Player Me 3D Avatar (Frontend)

The SignBridge mode displays a 3D avatar with lip-sync. To use your own:

1. Go to [readyplayer.me](https://readyplayer.me/)
2. Create a free account and design your avatar
3. Copy the avatar `.glb` URL
4. Append morph target params: `?morphTargets=viseme_aa,viseme_E,viseme_I,viseme_O,viseme_U,viseme_PP,viseme_FF,jawOpen&textureAtlas=1024`
5. Set `VITE_AVATAR_URL` in `frontend/.env`

## Architecture

```
main.py                    # Entry point — Agent, MCP tools, API endpoints
providers.py               # Multi-VLM fallback provider chain (5 adapters)
m5_bridge.py               # Camera bridge (RTSP/webcam → VideoForwarder)
processors/
├── signbridge_processor.py  # YOLO11 Pose + MediaPipe Hands → sign language
├── guidelens_processor.py   # YOLO11 Detection → hazard/object detection
├── ocr_processor.py         # On-demand OCR + dense scene description
└── mediapipe_hands.py       # Google MediaPipe 21-keypoint hand landmarker
mcp_tools/
├── maps_api.py              # Google Maps navigation (Directions + Places)
├── spatial_memory.py        # SQLite-backed spatial object memory
├── navigation_engine.py     # Smart navigation with dedup announcements
└── smart_tools.py           # Time, weather, emergency, color ID, device status
utils/
└── local_storage.py         # Frame store, detection cache, session management
tests/
├── test_day4.py             # Spatial memory + navigation engine tests
└── test_day5.py             # Haptic alerts + telemetry tests
```

## Vision Pipeline

### SignBridge Mode (Sign Language Translation)
```
Camera Frame
    ├──→ YOLO11 Pose (yolo11n-pose.pt) → 17 COCO body keypoints
    ├──→ MediaPipe Hands → 21 hand keypoints x2 → finger state analysis
    │        ├──→ Finger extended/curled detection
    │        └──→ Basic ASL letter recognition (A, B, D, I, L, V, W, Y, S, 5)
    ├──→ GestureBuffer (30-frame window) → gesture classification
    ├──→ GlossTranslator → HuggingFace NLP or rule-based fallback
    └──→ Skeleton + Hand overlay → WebRTC published video
```

### GuideLens Mode (Environmental Awareness)
```
Camera Frame
    ├──→ YOLO11 Detection (yolo11n.pt) → 80 COCO object classes
    ├──→ BboxTracker → approach speed (growth rate estimation)
    ├──→ NavigationEngine → smart dedup announcements
    ├──→ SpatialMemory (SQLite) → "what was that object earlier?"
    ├──→ Auto-haptic alerts (critical/warning/caution from bbox growth)
    └──→ Bbox + hazard overlay → WebRTC published video
```

### Both Modes
```
    ├──→ OCR Processor → multi-VLM text extraction (periodic + on-demand)
    ├──→ Gemini 2.5 Flash Realtime → voice conversation + vision reasoning
    └──→ MCP Tools → navigation directions, object memory recall
```

## API Endpoints

### Core
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/mode` | Current agent mode |
| `POST` | `/switch-mode` | Toggle SignBridge ↔ GuideLens |
| `POST` | `/set-mode/{mode}` | Set specific mode |
| `GET` | `/token?user_id=` | Generate Stream user token |
| `GET` | `/stream-config` | Stream API key for frontend |
| `GET` | `/transcript` | Conversation transcript |
| `DELETE` | `/transcript` | Clear transcript |

### Providers
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/providers` | Status of all VLM providers |
| `POST` | `/providers/preferred/{id}` | Set preferred provider |
| `GET` | `/providers/fallback-events` | Fallback event log |

### OCR / VLM
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/ocr-results` | Cached OCR text results |
| `POST` | `/ocr/read` | Trigger OCR on current frame |
| `POST` | `/ocr/describe` | Trigger dense scene description |

### Telemetry & Navigation (Day 5)
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/telemetry` | Real-time aggregated metrics |
| `GET` | `/navigation/hazards` | Active hazard alerts |
| `GET` | `/navigation/hazards/poll` | Consumable hazard alerts |
| `GET` | `/navigation/environment` | Environment summary |

### Smart Tools (Day 5)
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/time` | Current time and date |
| `GET` | `/weather?location=` | Weather for a location |
| `GET` | `/device-status` | Device status info |
| `GET` | `/emergencies` | Emergency alert log |
| `POST` | `/emergency` | Manually trigger emergency |

### Spatial Memory (Day 4)
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/memory/search` | Search spatial memory |
| `GET` | `/memory/recent` | Recent detections |
| `GET` | `/memory/summary` | Memory statistics |

## MCP Tools (Available to Gemini)

| Tool | Description |
|------|-------------|
| `read_text_in_scene` | OCR text extraction from video |
| `describe_scene_detailed` | Dense scene description via VLM |
| `get_walking_directions` | Google Maps walking navigation |
| `search_nearby_places` | Find nearby points of interest |
| `remember_object` | Log object to spatial memory |
| `recall_objects` | Query spatial memory |
| `get_environment_context` | Current scene understanding |
| `trigger_haptic_alert` | Send hazard alert to frontend |
| `get_time_and_date` | Current local time, date, day of week |
| `get_weather` | Weather conditions via Open-Meteo (free, no key) |
| `identify_colors` | Color identification using Gemini VLM |
| `emergency_alert` | Trigger emergency alert + log |
| `get_device_status` | Battery level, camera/mic status, uptime |

## VLM Provider Chain

The multi-provider fallback chain (configured in `providers.py`):

```
Gemini Pro Vision → xAI Grok → Azure GPT-4o → NVIDIA Cosmos 2 → HuggingFace
      (primary)       (fast)      (reliable)      (dense desc)     (free tier)
```

Each provider uses lazy initialization, health checks, and cooldowns. If one fails, the chain automatically tries the next.

## Running Tests

```bash
# All tests
uv run pytest tests/ -v

# Day 4 tests (spatial memory, navigation)
uv run pytest tests/test_day4.py -v

# Day 5 tests (haptic alerts, telemetry)
uv run pytest tests/test_day5.py -v
```
