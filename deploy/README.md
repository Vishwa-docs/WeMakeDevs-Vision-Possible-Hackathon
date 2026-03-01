# WorldLens — Docker Deployment

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/YOUR_USERNAME/worldlens.git
cd worldlens

# Copy the env template and fill in your API keys
cp deploy/.env.docker .env
```

Edit `.env` and add your API keys:
- **STREAM_API_KEY** + **STREAM_API_SECRET** — Get from [GetStream Dashboard](https://getstream.io/dashboard/)
- **GOOGLE_API_KEY** — Get from [Google AI Studio](https://aistudio.google.com/apikey)
- **MAPS_API_KEY** (optional) — Get from [Google Cloud Console](https://console.cloud.google.com/apis/credentials)

### 2. Build and run

```bash
# Build and start (first time takes ~5 min)
docker compose -f deploy/docker-compose.yml up --build

# Or run in background
docker compose -f deploy/docker-compose.yml up --build -d
```

### 3. Open the app

- **Backend API**: http://localhost:8000
- **Frontend**: Open `frontend/index.html` or run the frontend dev server separately

### 4. Stop

```bash
docker compose -f deploy/docker-compose.yml down
```

---

## Pull from Docker Hub (pre-built)

If the image is published:

```bash
# Pull the image
docker pull YOUR_DOCKERHUB_USERNAME/worldlens:latest

# Run with your API keys
docker run -d \
  --name worldlens \
  -p 8000:8000 \
  --env-file .env \
  YOUR_DOCKERHUB_USERNAME/worldlens:latest
```

## Build and Push to Docker Hub

```bash
# Build
docker build -t YOUR_DOCKERHUB_USERNAME/worldlens:latest -f deploy/Dockerfile .

# Push
docker push YOUR_DOCKERHUB_USERNAME/worldlens:latest
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `STREAM_API_KEY` | Yes | GetStream API key for WebRTC |
| `STREAM_API_SECRET` | Yes | GetStream API secret |
| `GOOGLE_API_KEY` | Yes | Gemini LLM key |
| `MAPS_API_KEY` | No | Google Maps for navigation |
| `HF_API_TOKEN` | No | HuggingFace for NLP |
| `AGENT_MODE` | No | `guidelens` (default) or `signbridge` |

## Troubleshooting

- **Port 8000 in use**: Change the port mapping in `docker-compose.yml`
- **Out of memory**: YOLO models need ~2GB RAM minimum
- **No camera**: The app uses your browser webcam via WebRTC, not Docker's
