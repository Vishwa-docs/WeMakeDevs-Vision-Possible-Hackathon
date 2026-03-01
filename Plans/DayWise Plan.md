# WorldLens — Day-by-Day Implementation Plan

## Day 1: Infrastructure & "Hello World" Skeletons

### Backend (Completed)
- [x] Initialized `uv` Python 3.12 project in `/backend`
- [x] Installed Vision Agents SDK with all plugins: `getstream`, `gemini`, `ultralytics`, `moondream`, `nvidia`, `huggingface`
- [x] Created `main.py` — Core agent with `gemini.Realtime(fps=5)` + `getstream.Edge()`
  - Dual-mode system prompts (SignBridge / GuideLens)
  - Event logging for user/agent speech transcripts
  - MCP tool stubs for `get_walking_directions` and `search_memory`
  - `AgentLauncher` with server mode (CORS configured for frontend)
  - Custom `/mode` endpoint for frontend to check active mode
- [x] Created `m5_bridge.py` — Camera bridge with webcam fallback
  - RTSP/MJPEG/webcam capture via OpenCV
  - Async capture loop with configurable FPS
  - `FrameStore` in-memory frame buffer with optional disk persistence
- [x] Created processor scaffolds:
  - `processors/signbridge_processor.py` — `VideoProcessorPublisher` with custom events (`SignDetectedEvent`, `SignTranslationEvent`)
  - `processors/guidelens_processor.py` — `VideoProcessorPublisher` with hazard detection events, direction/distance estimation
- [x] Created MCP tool scaffolds:
  - `mcp_tools/maps_api.py` — Google Maps Directions API integration (stub with fallback)
  - `mcp_tools/spatial_memory.py` — In-memory spatial object memory with search
- [x] Created `utils/local_storage.py`:
  - `LocalFrameStore` — On-disk frame persistence with automatic eviction
  - `DetectionCache` — Perceptual-hash LRU cache for detection deduplication
  - `SessionManager` — Session state tracking and crash recovery
- [x] Created `.env.example` with all required API keys
- [x] Created `requirements.txt` for pip-based installs

### Frontend (Completed)
- [x] Scaffolded Vite + React 18 + TypeScript in `/frontend`
- [x] Installed `@stream-io/video-react-sdk` for WebRTC video calls
- [x] Created `VideoRoom.tsx` — Full Stream Video integration:
  - `StreamVideoClient` initialization with API key
  - Call join/leave lifecycle
  - `ParticipantView` for local + remote video
  - Camera and microphone enable on join
- [x] Created `StatusBar.tsx` — Backend health + mode indicator
- [x] Created `ChatLog.tsx` — Transcript history with auto-scroll
- [x] Created `TelemetryPanel.tsx` — Metrics display (latency, FPS, VLM info)
- [x] Created `AlertOverlay.tsx` — Hazard warning flash overlay (mock haptics scaffold)
- [x] Created `useAgentSession` hook — Session management with backend API
- [x] Created `utils/api.ts` — Backend API client
- [x] Created `types/index.ts` — Full TypeScript interfaces
- [x] Dark theme UI with WorldLens branding (app.css, index.css)
- [x] Created `.env.example`

### Day 1 Goal:
> Open the React app on localhost → see webcam → have a voice conversation with the Gemini Agent via Stream Edge.

---

## Day 2: Concurrent Vision Processors & Core Intelligence

### Backend (Completed)
- [x] **SignBridge:** Custom `SignBridgeProcessor` (VideoProcessorPublisher) using `ultralytics` YOLO11 Pose
  - Extracts 17 COCO skeletal keypoints per person  
  - Draws upper-body focused skeleton overlay with wrist highlights
  - `GestureBuffer` with temporal motion analysis (30-frame window)
  - Rule-based gesture classifier (WAVE, RAISE-HAND, POINT, ACTIVE-SIGN)
  - `GlossTranslator` with optional HuggingFace Inference API for gloss→English
  - Custom events: `SignDetectedEvent`, `GestureBufferEvent`, `SignTranslationEvent`
  - Async YOLO inference via thread-pool executor + mutex to prevent frame queueing
- [x] **GuideLens:** Custom `GuideLensProcessor` (VideoProcessorPublisher) using `ultralytics` YOLO11 Detection
  - Detects 80 COCO classes, filters for hazard-relevant objects (person, car, bicycle, truck, etc.)
  - Real-time bounding-box drawing with class labels and distance estimates
  - `BboxTracker` — per-class area history for approach-speed computation
  - Direction estimation (left/centre/right) from bbox centre position
  - Distance estimation (near/medium/far) from bbox area ratio
  - `HazardDetectedEvent` emitted when objects are near or approaching fast
  - `SceneSummaryEvent` emitted periodically with object counts
  - Detection log for spatial-memory integration (Day 4)
- [x] Updated `main.py`:
  - `_build_processors()` factory instantiates processors based on `AGENT_MODE`
  - Enhanced system prompts describing processor capabilities to the LLM
  - Event subscriptions for all processor events with structured logging
  - POST `/switch-mode` endpoint to toggle SignBridge ↔ GuideLens
  - POST `/set-mode/{mode}` endpoint for explicit mode setting
  - Processors wired into `Agent(processors=[...])` constructor

### Frontend (Completed)
- [x] Mode toggle button in `StatusBar` with icon + label + switch indicator
  - Calls POST `/switch-mode` on backend
  - Shows notification banner: "Mode will change on next session"
- [x] `switchMode` + `setMode` API functions in `utils/api.ts`
- [x] `toggleMode` function in `useAgentSession` hook
- [x] Mode notification banner in `App.tsx` with animation
- [x] Updated telemetry display showing active processor (YOLO-Pose / YOLO-Detect)

### Architecture Decisions:
- Used YOLO Detection (`yolo11n.pt`) instead of Moondream CloudDetectionProcessor for GuideLens: faster inference, no API key required, works offline
- Mode switch takes effect on next session (processors are set at Agent creation time per SDK architecture)
- Both processors run YOLO inference in thread-pool executors to avoid blocking the event loop
- Custom events inherit from `vision_agents.core.events.BaseEvent` with proper `type` field defaults

### Day 2 Goal:
> Both processors running real CV pipelines (YOLO Pose + YOLO Detection) with skeleton/bbox overlays, event emission, and mode toggle from the frontend.

---

## Day 3: Advanced Visuals — OCR, NVIDIA VLM, & 3D Avatar

### Backend (Completed)
- [x] **OCR Processor:** Created `processors/ocr_processor.py` (VideoProcessor)
  - Captures frames at 1 FPS into a rolling buffer for on-demand VLM queries
  - `read_text()` method sends current frame to provider_manager with OCR-focused prompt
  - `describe_scene()` method generates dense scene descriptions via NVIDIA/Gemini VLM
  - Background periodic OCR scan loop (every 20s) caches visible text
  - Custom events: `OCRResultEvent`, `SceneDescriptionEvent`
  - Results cached in deque for frontend polling
- [x] **MCP Tools:** Registered `read_text_in_scene` and `describe_scene_detailed` function tools
  - Gemini Realtime can autonomously call these when user asks about text/signs
  - Uses multi-provider fallback chain: Gemini → Grok → Azure → NVIDIA → HuggingFace
- [x] **API Endpoints:** Added `/ocr-results`, `/ocr/read`, `/ocr/describe`
  - Frontend can poll OCR results for overlay display
  - Manual trigger endpoints for testing
- [x] **System Prompt Update:** GuideLens instructions now describe OCR + VLM tools
- [x] **Both modes get OCR:** OCRProcessor runs alongside YOLO in both SignBridge and GuideLens

### Frontend (Completed)
- [x] **3D Avatar:** `components/Avatar3D/Avatar3D.tsx` using React Three Fiber
  - Loads Ready Player Me `.glb` model via `@react-three/drei` useGLTF
  - Multi-frequency jaw morph target animation (viseme_aa, jawOpen)
  - Secondary viseme animations (viseme_E, I, O, U, PP, FF) for realism
  - Smooth interpolation: quick open, slower close
  - Environment preset for realistic lighting/reflections
  - Speaking indicator bar animation at bottom
  - Renders in SignBridge mode sidebar with header showing speaking state
- [x] **OCR Overlay:** `components/OCROverlay.tsx`
  - Polls `/ocr-results` endpoint at configurable interval
  - Renders detected text as floating overlay on video feed
  - Results fade out after configurable display duration
  - Shows provider and age for each detection
- [x] **API Functions:** Added `getOCRResults`, `triggerOCRRead`, `triggerSceneDescription` to `utils/api.ts`
- [x] **App Integration:** Updated `App.tsx`
  - Avatar3D renders in sidebar during SignBridge mode
  - OCR overlay renders over video area during active sessions
  - Agent speaking state detected from transcript polling → drives avatar lip-sync
  - Telemetry updated to show OCR processor count
- [x] **Dependencies:** Installed `three`, `@react-three/fiber`, `@react-three/drei`, `@types/three`

### Architecture Decisions:
- Used `provider_manager` fallback chain for OCR instead of Moondream CloudVLM: more flexible, supports 5 VLM providers with automatic failover
- OCR runs on-demand via MCP tools (not every frame) to avoid excessive API costs
- Background periodic scan (20s) catches text passively without overwhelming providers
- Avatar lip-sync driven by transcript activity (practical for hackathon; real audio-frequency driving would require WebRTC audio stream access)

### Day 3 Goal:
> OCR reads text from camera frames via VLM, NVIDIA/Gemini provides dense scene descriptions, 3D avatar lip-syncs to agent speech, and OCR overlay displays detected text on the video feed.

---

## Day 4: Agentic Tool Calling & Spatial Memory (MCP)

### Backend (Completed)
- [x] Wired `get_walking_directions` MCP tool to live Google Maps Directions API
  - `maps_api.py` upgraded: geocoding, directions, places search, geolocation
  - Graceful stub fallback when `MAPS_API_KEY` is not set
- [x] Migrated `SpatialMemory` from in-memory to async SQLite (`spatial_memory.db`)
  - `aiosqlite` used for async database operations
  - Background loop logging YOLO detections with auto-eviction (30-minute TTL)
  - `search_memory` tool returns time-ago formatted results with location context
- [x] Created `NavigationEngine` in `mcp_tools/navigation_engine.py`
  - Route state management (start, waypoints, progress)
  - Hazard alert queue with severity levels (INFO, WARNING, DANGER, CRITICAL)
  - `pop_hazard_alerts(since)` for frontend polling
- [x] **Google MediaPipe Integration** — `processors/mediapipe_hands.py`
  - MediaPipe Hand Landmarker with 21 keypoints per hand
  - `hand_landmarker.task` model file (7.5 MB)
  - Finger state analysis (extended/curled per finger)
  - ASL static finger-spelling recognition: A, B, D, I, L, V, W, Y, S, 5
  - Fingertip circle annotations (green=extended, red=curled) + handedness labels
  - Lazy import with graceful fallback if MediaPipe unavailable
- [x] **SignBridge Processor Enhanced** — dual YOLO + MediaPipe pipeline
  - `SignDetectedEvent` expanded with `num_hands`, `finger_states`, `asl_letters`
  - Gesture classifier enriched with ASL letter context
  - Telemetry includes `total_hands_detected`, `total_asl_letters_detected`
- [x] Multi-round tool calling tested with `max_tool_rounds=5`
- [x] 14 Day 4 backend tests passing (spatial memory CRUD, maps API, navigation engine)

---

## Day 5: Synthesis & Haptic UI

### Backend (Completed)
- [x] `trigger_haptic_alert` MCP tool — LLM detects approaching objects via bbox growth rate
  - Severity mapping: CRITICAL→strong, DANGER→medium, WARNING→light, INFO→gentle
  - Callable by agent during realtime conversations
- [x] Real telemetry metrics endpoint (`GET /telemetry`)
  - Per-processor telemetry: model, device, FPS, frame counts, detection counts
  - Aggregate telemetry: total_frames_processed, total_objects_detected, total_hazards_detected, total_gestures_detected
  - Active mode and provider chain status
- [x] GuideLens processor auto-haptic: triggers alert when near + approaching bbox detected
- [x] 10 Day 5 backend tests passing (haptic alerts, processor telemetry, severity mapping)
- [x] All 24 backend tests pass (`pytest tests/ -v`)
- [x] 5 new MCP smart tools (`mcp_tools/smart_tools.py`):
  - `get_time_and_date` — local time, date, day of week with spoken summary
  - `get_weather` — Open-Meteo API (free, no key), WMO weather codes, temp/humidity/wind
  - `identify_colors` — instructs agent to use Gemini VLM for color description
  - `emergency_alert` — logs emergency, would notify contacts in production
  - `get_device_status` — battery, camera/mic, uptime (simulated for demo)
- [x] 5 new API endpoints: `/time`, `/weather`, `/device-status`, `/emergencies`, `/emergency`
- [x] Total: 12 MCP tools registered with the agent

### Frontend (Completed)
- [x] `AlertOverlay` v2 — fully animated hazard warnings
  - Severity levels (info/warning/danger/critical) with distinct colors
  - Directional glow indicators (left/centre/right edge glow)
  - Web Audio API chime on each alert (440Hz sine)
  - Vibration API support for mobile devices
  - Auto-dismiss with configurable timeouts per severity
  - Slide-in/fade-out CSS animations
- [x] Enterprise telemetry display (`TelemetryPanel.tsx` rewrite)
  - Inference latency gauge, FPS gauge, detection counters
  - Provider chain status with color indicators
  - Hazard severity breakdown
- [x] `pollHazardAlerts()` — 2-second polling interval with timestamp tracking
- [x] `getTelemetry()` — 3-second polling for real metrics
- [x] TypeScript interfaces expanded: `AggregateTelemetry`, `HazardAlert`, `TelemetryData`
- [x] Clean TypeScript + Vite build verified
- [x] Major UI overhaul — glassmorphism dark theme:
  - CSS custom properties for consistent design tokens
  - Glassmorphism surfaces with backdrop-filter blur
  - Ambient animated gradient background on landing page
  - Floating hero icon with animation
  - Mode cards (GuideLens / SignBridge) with active state highlight
  - Feature badges showing tech stack (Gemini, YOLO, MediaPipe, etc.)
  - Sidebar section headers with conversation count badge
  - Live telemetry indicator with pulsing dot
  - Custom scrollbar styling throughout
  - Improved chat message animations (slide-in)
  - Button spinner for loading state
  - Responsive breakpoints (768px, 480px)
- [x] Fixed 10 CSS/component bugs:
  - Toast class mismatch (`toast-${type}` → `${type}`)
  - Toast dismiss button class (`toast-close` → `toast-dismiss`)
  - ProviderSelector 5 className mismatches fixed to match CSS
  - Provider status dot `healthy` → `ready`
  - Removed dead Alert V1 CSS
  - Added `.provider-dropdown-footer` styles

### Documentation (Completed)
- [x] Backend `.env.example` — comprehensive with setup URLs for every service
- [x] Frontend `.env.example` — added `VITE_AVATAR_URL` with Ready Player Me guide
- [x] `backend/README.md` — full rewrite: API keys setup, architecture, vision pipelines, endpoints, 12 MCP tools
- [x] `frontend/README.md` — full rewrite: project-specific docs, setup guide, structure, tech stack
- [x] Root `README.md` — updated architecture diagram, MediaPipe in tech stack, 12 MCP tools, Day 5 progress
- [x] `requirements.txt` — added `mediapipe>=0.10.20`
- [x] Avatar3D configurable via `VITE_AVATAR_URL` environment variable

---

## Day 6: Errors, Demo, Production Readiness

---

## Day 7: Hardware Port & Live Testing

- [ ] Connect M5StickC Plus via RTSP to `m5_bridge.py`
- [ ] Pipe OpenCV frames into Vision Agents `VideoForwarder`
- [ ] Live room walk test
- [ ] Fallback: tethered laptop webcam for live Q&A
