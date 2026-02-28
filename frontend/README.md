# WorldLens Frontend

React 19 + TypeScript + Vite frontend for the WorldLens assistive vision platform.

## Prerequisites

- **Node.js 18+** (recommended: 20 LTS)
- **npm** (comes with Node.js)
- Backend server running at `http://localhost:8000`

## Quick Start

```bash
# 1. Install dependencies
npm install

# 2. Copy and configure environment
cp .env.example .env
# Edit .env — at minimum set VITE_STREAM_API_KEY

# 3. Start development server
npm run dev
# Opens at http://localhost:5173
```

## Environment Variables

| Variable | Required | Description | Where to Get |
|----------|----------|-------------|--------------|
| `VITE_STREAM_API_KEY` | ✅ | GetStream API key (same as backend `STREAM_API_KEY`) | [getstream.io/dashboard](https://getstream.io/dashboard/) |
| `VITE_BACKEND_URL` | ✅ | Backend server URL | Default: `http://localhost:8000` |
| `VITE_AVATAR_URL` | Optional | Custom `.glb` avatar URL with viseme morph targets | Any rigged GLB model |

### 3D Avatar (SignBridge)

The SignBridge mode shows a 3D avatar with lip-sync. A built-in geometric
fallback avatar is used by default. If you have your own `.glb` model with
viseme morph targets (`viseme_aa`, `viseme_E`, etc.), set `VITE_AVATAR_URL`
in your `.env`.

> **Note:** Ready Player Me was discontinued on January 31, 2026. Their hosted model URLs no longer work. Please use a custom .gib model (OPTIONAL)

## Project Structure

```
src/
├── App.tsx              # Main app — session management, polling, layout
├── App.css              # Dark theme styling
├── main.tsx             # React entry point
├── components/
│   ├── VideoRoom.tsx    # Stream Video SDK — WebRTC call
│   ├── StatusBar.tsx    # Connection status + mode toggle
│   ├── ChatLog.tsx      # Conversation transcript history
│   ├── TelemetryPanel.tsx  # Real-time metrics dashboard
│   ├── AlertOverlay.tsx    # Hazard warnings (audio + visual + haptic)
│   ├── Avatar3D/           # 3D avatar lip-sync (React Three Fiber)
│   ├── OCROverlay.tsx      # Text detection overlay on video
│   ├── ProviderSelector.tsx # VLM provider management
│   └── Toast.tsx           # Toast notification system
├── hooks/
│   └── useAgentSession.ts  # Session lifecycle hook
├── types/
│   └── index.ts            # TypeScript interfaces
└── utils/
    └── api.ts              # Backend API client
```

## Key Features

- **WebRTC Video/Audio** via Stream Video React SDK
- **3D Avatar** with lip-sync morph targets (React Three Fiber + Ready Player Me)
- **Hazard Alert Overlay** with severity levels, directional glow, Web Audio chimes
- **Real-time Telemetry** — inference latency, object counts, provider chain status
- **OCR Text Overlay** — detected text displayed over video feed
- **Multi-VLM Provider Panel** — switch providers, see fallback events
- **Dark theme UI** with responsive layout

## Available Scripts

```bash
npm run dev        # Start dev server with HMR
npm run build      # Production build to dist/
npm run preview    # Preview production build
npm run lint       # Run ESLint
```

## Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| React | 19 | UI framework |
| Vite | 7 | Build tool |
| TypeScript | 5.9 | Type safety |
| `@stream-io/video-react-sdk` | Latest | WebRTC video calls |
| `@react-three/fiber` | Latest | Three.js React bindings |
| `@react-three/drei` | Latest | Three.js helpers |
| `three` | Latest | 3D rendering engine |
