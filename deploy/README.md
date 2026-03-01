# WorldLens — Docker Deployment

Single-container Docker deployment for WorldLens. The image includes the Python backend (Vision Agents SDK + YOLO11 + MediaPipe + Gemini Realtime) and a pre-built React frontend served as static files.

> **Note:** The Docker image must be built with `--platform linux/amd64` because the `mediapipe` Python package does not provide ARM64 Linux wheels. On Apple Silicon Macs, Docker runs under Rosetta emulation.

---

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/Vishwa-docs/WeMakeDevs-Vision-Possible-Hackathon.git
cd WeMakeDevs-Vision-Possible-Hackathon

# Copy the env template
cp deploy/.env.docker .env
```

Edit `.env` and add your API keys:

| Variable | Required | How to Get |
|----------|----------|------------|
| `STREAM_API_KEY` | Yes | [getstream.io/dashboard](https://getstream.io/dashboard/) |
| `STREAM_API_SECRET` | Yes | Same dashboard |
| `GOOGLE_API_KEY` | Yes | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `MAPS_API_KEY` | No | [console.cloud.google.com](https://console.cloud.google.com/apis/credentials) |
| `HF_API_TOKEN` | No | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `NGC_API_KEY` | No | [NVIDIA NGC](https://org.ngc.nvidia.com/setup/api-key) |
| `XAI_API_KEY` | No | [console.x.ai](https://console.x.ai/) |
| `AGENT_MODE` | No | `guidelens` (default) or `signbridge` |

### 2. Build and run

```bash
# Using docker-compose (recommended)
docker compose -f deploy/docker-compose.yml up --build

# Or run in background
docker compose -f deploy/docker-compose.yml up --build -d

# Or build manually
docker build --platform linux/amd64 -f deploy/Dockerfile -t worldlens:latest .
docker run --platform linux/amd64 -p 8000:8000 --env-file .env worldlens:latest
```

### 3. Open the app

Once running, everything is served from a single port:

- **Frontend + Backend**: http://localhost:8000
- **Health check**: http://localhost:8000/health
- **API docs**: See [backend/README.md](../backend/README.md) for all endpoints

### 4. Stop

```bash
docker compose -f deploy/docker-compose.yml down
```

---

## How the Docker Build Works

The Dockerfile uses a **two-stage multi-stage build**:

### Stage 1: Frontend Build (Node.js 22)
```
node:22-slim
├── npm ci                          # Install dependencies
├── VITE_BACKEND_URL=""             # Empty = same-origin (works in container)
├── VITE_STREAM_API_KEY from env    # Baked into the JS bundle
└── npm run build                   # → dist/ folder with static files
```

### Stage 2: Python Runtime
```
python:3.12-slim (linux/amd64)
├── uv sync                         # Install Python dependencies
├── COPY dist/ → /app/static/       # Frontend files served by FastAPI
├── COPY yolo11n.pt, hand_landmarker.task  # ML models
└── uv run main.py serve --host 0.0.0.0 --port 8000
```

The backend's FastAPI serves the frontend static files at `/` with SPA catch-all routing.

---

## Build Script

A convenience script is provided:

```bash
./deploy/build.sh
```

This runs `docker build` with the correct `--platform` flag and tags the image as `worldlens:latest`.

---

## Environment Variables (Full Reference)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `STREAM_API_KEY` | Yes | — | GetStream API key for WebRTC transport |
| `STREAM_API_SECRET` | Yes | — | GetStream API secret |
| `GOOGLE_API_KEY` | Yes | — | Google Gemini 2.5 Flash Realtime LLM |
| `MAPS_API_KEY` | No | — | Google Maps (Directions, Geocoding, Places, Geolocation) |
| `HF_API_TOKEN` | No | — | HuggingFace NLP for SignBridge gloss → English |
| `NGC_API_KEY` | No | — | NVIDIA Cosmos 2 VLM for dense scene descriptions |
| `XAI_API_KEY` | No | — | xAI Grok Vision for OCR fallback |
| `AZURE_OPENAI_ENDPOINT` | No | — | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_KEY` | No | — | Azure OpenAI API key |
| `AZURE_OPENAI_DEPLOYMENT` | No | — | Azure OpenAI deployment name |
| `AGENT_MODE` | No | `guidelens` | Starting mode: `guidelens` or `signbridge` |
| `VITE_STREAM_API_KEY` | Build-time | — | Frontend Stream key (set during `docker build`) |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **Port 8000 in use** | `lsof -ti:8000 \| xargs kill -9` or change the port mapping in `docker-compose.yml` |
| **Out of memory** | YOLO models + Gemini client need ~2 GB RAM minimum. Increase Docker memory limit |
| **No camera access** | The app uses your **browser's webcam** via WebRTC, not Docker's. Camera access is browser-side |
| **mediapipe build fails** | Ensure `--platform linux/amd64` is set. mediapipe has no ARM64 Linux wheel |
| **Stream API error** | Make sure `--env-file .env` is passed to `docker run`. Without it, environment variables are empty |
| **Frontend shows blank page** | Check that `VITE_STREAM_API_KEY` was set as a build arg. The frontend needs it baked into the JS bundle |

---

## Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 2 GB | 4 GB |
| CPU | 2 cores | 4 cores |
| Disk | 3 GB (image size) | 5 GB |
| Network | Required | Required (all features need internet) |
