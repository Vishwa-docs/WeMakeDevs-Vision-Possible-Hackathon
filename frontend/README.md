# WorldLens Frontend

The React 19 frontend for WorldLens — a real-time assistive vision platform. Connects to the backend via WebRTC (GetStream Video SDK) for live video/audio and REST polling for hazard alerts, telemetry, OCR results, and navigation status.

---

## Prerequisites

- **Node.js 18+** (recommended: 20 LTS)
- **npm** (comes with Node.js)
- Backend server running at `http://localhost:8000` (see [backend/README.md](../backend/README.md))
- **Active internet connection** — required for GetStream WebRTC

## Quick Start

```bash
# 1. Install dependencies
npm install

# 2. Configure environment
cp .env.example .env
# Set VITE_STREAM_API_KEY (same as backend's STREAM_API_KEY)
# Set VITE_BACKEND_URL=http://localhost:8000

# 3. Start development server
npm run dev
# Opens at http://localhost:5173
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `VITE_STREAM_API_KEY` | Yes | — | GetStream API key (same value as backend `STREAM_API_KEY`) |
| `VITE_BACKEND_URL` | Yes | `http://localhost:8000` | Backend server URL |
| `VITE_AVATAR_URL` | No | Built-in geometric | Custom `.glb` avatar URL with viseme morph targets |
| `VITE_BASE_PATH` | No | `/` | Base path for GitHub Pages or subdirectory deployment |

---

## Project Structure

```
src/
├── App.tsx                    # Main app — session lifecycle, polling, layout
├── App.css                    # Dark glassmorphism theme
├── main.tsx                   # React entry point
│
├── components/
│   ├── VideoRoom.tsx          # Stream Video SDK — WebRTC call (video + audio)
│   ├── StatusBar.tsx          # Connection status + GuideLens/SignBridge mode toggle
│   ├── ChatLog.tsx            # Real-time conversation transcript history
│   ├── AlertOverlay.tsx       # Hazard warnings — severity levels, directional glow,
│   │                          #   Web Audio chimes, haptic vibration API
│   ├── TelemetryPanel.tsx     # Enterprise metrics — inference latency, FPS,
│   │                          #   object counts, provider chain status
│   ├── OCROverlay.tsx         # Detected text overlay on video feed
│   ├── ProviderSelector.tsx   # VLM provider management — switch, view fallback events
│   ├── Toast.tsx              # Toast notification system
│   └── Avatar3D/              # 3D avatar with lip-sync morph targets
│       ├── AvatarScene.tsx    # Three.js scene setup
│       ├── AvatarModel.tsx    # GLB model loader + viseme animation
│       └── FallbackAvatar.tsx # Built-in geometric avatar (no external model needed)
│
├── hooks/
│   └── useAgentSession.ts     # Session lifecycle hook — token generation,
│                               #   call creation, agent joining, cleanup
│
├── types/
│   └── index.ts               # TypeScript interfaces — AgentStatus, HazardAlert,
│                               #   OCRResult, TelemetryData, TranscriptEntry, etc.
│
└── utils/
    └── api.ts                 # Backend API client — all REST endpoint wrappers
                               #   /health, /mode, /token, /transcript, /telemetry,
                               #   /navigation/hazards/poll, /ocr-results, etc.
```

---

## Key Features

### WebRTC Video/Audio
- Full-duplex video and audio via **GetStream Video React SDK**
- User's camera feed streams to the backend for YOLO processing
- Agent's annotated video (bounding boxes, skeleton overlays) streams back

### Hazard Alert System (`AlertOverlay.tsx`)
- **3 severity levels:** Critical (red pulse), Warning (amber), Caution (blue)
- **Directional glow:** Left, center, or right edge glow based on hazard direction
- **Web Audio chimes:** Different tones per severity level
- **Haptic vibration:** Uses the Vibration API on supported devices
- Polls `/navigation/hazards/poll` every 2 seconds

### Real-time Telemetry (`TelemetryPanel.tsx`)
- Inference latency (ms per frame)
- Object detection counts
- Active VLM provider status
- Session uptime and frame rate
- Polls `/telemetry` every 3 seconds

### OCR Text Overlay (`OCROverlay.tsx`)
- Displays detected text over the video feed
- Shows provider source and timestamp
- Polls `/ocr-results` every 5 seconds

### 3D Avatar (`Avatar3D/`)
- Lip-sync via viseme morph targets (`viseme_aa`, `viseme_E`, etc.)
- Built-in geometric fallback — no external model required
- Optional custom `.glb` model via `VITE_AVATAR_URL`
- Used primarily in SignBridge mode

### Mode Switching
- Toggle between GuideLens and SignBridge via the StatusBar
- Calls `POST /switch-mode` on the backend
- UI adapts components based on active mode

---

## Available Scripts

```bash
npm run dev        # Start dev server with HMR (port 5173)
npm run build      # Production build to dist/
npm run preview    # Preview production build locally
npm run lint       # Run ESLint
```

---

## Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| React | 19.2 | UI framework |
| Vite | 7.3 | Build tool with HMR |
| TypeScript | 5.9 | Type safety |
| `@stream-io/video-react-sdk` | 1.32+ | WebRTC video calls via GetStream |
| `@react-three/fiber` | 9.5+ | Three.js React bindings (3D avatar) |
| `@react-three/drei` | 10.7+ | Three.js helpers (orbit controls, loaders) |
| `three` | 0.183+ | 3D rendering engine |

### Dev Dependencies

| Library | Purpose |
|---------|---------|
| `vitest` | Test runner (46 tests) |
| `@testing-library/react` | React component testing |
| `@testing-library/jest-dom` | DOM assertion matchers |
| `jsdom` | Browser environment for tests |
| `eslint` + `typescript-eslint` | Linting |

---

## Testing

```bash
# Run all 46 tests
npx vitest run --reporter verbose

# Watch mode
npx vitest
```

Tests cover:
- Component rendering (StatusBar, ChatLog, AlertOverlay, TelemetryPanel)
- Session lifecycle hooks
- API client functions
- Mode switching behavior
- Alert severity and direction logic

---

## Docker Deployment

When deployed via Docker (see [deploy/README.md](../deploy/README.md)), the frontend is pre-built during the Docker image build and served as static files by the backend's FastAPI server at `http://localhost:8000/`. No separate frontend server is needed in production.
