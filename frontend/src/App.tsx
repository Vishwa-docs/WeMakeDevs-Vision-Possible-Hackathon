/**
 * WorldLens — Main Application
 * Day 1: Core layout with Stream Video integration, status bar, chat log.
 */
import { useState, useCallback, useEffect } from "react";
import { useAgentSession } from "./hooks/useAgentSession";
import { StatusBar } from "./components/StatusBar";
import { VideoRoom } from "./components/VideoRoom";
import { ChatLog } from "./components/ChatLog";
import { TelemetryPanel } from "./components/TelemetryPanel";
import { AlertOverlay } from "./components/AlertOverlay";
import type { TranscriptEntry, TelemetryData } from "./types";
import "./App.css";

function App() {
  const { session, status, loading, error, startSession, stopSession } =
    useAgentSession();

  const [callId, setCallId] = useState(`worldlens-${Date.now()}`);
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  // TODO (Day 3): Populate transcript from real-time transcription events via WebSocket/SSE
  // TODO (Day 5): Drive alertActive from hazard detection events emitted by GuideLens processor
  const alertActive = false; // placeholder — will become useState once hazard events are wired up
  const [startTime] = useState(Date.now());
  const [uptime, setUptime] = useState(0);

  // Update uptime counter
  useEffect(() => {
    if (!session) return;
    const interval = setInterval(() => {
      setUptime((Date.now() - startTime) / 1000);
    }, 1000);
    return () => clearInterval(interval);
  }, [session, startTime]);

  // Mock telemetry (Day 5: real metrics from backend)
  const telemetry: TelemetryData = {
    edgeLatency: 24,
    activeVLM: status.mode === "signbridge" ? "YOLO-Pose" : "Gemini-2.5-Flash",
    fps: 5,
    processorCount: status.connected ? 1 : 0,
    uptime,
  };

  const handleStart = useCallback(async () => {
    const id = `worldlens-${Date.now()}`;
    setCallId(id);
    await startSession(id);
  }, [startSession]);

  const handleStop = useCallback(async () => {
    await stopSession();
    setTranscript([]);
  }, [stopSession]);

  return (
    <div className="app">
      {/* Alert overlay for hazard warnings */}
      <AlertOverlay active={alertActive} message="Obstacle approaching!" />

      {/* Header */}
      <header className="app-header">
        <div className="logo">
          <h1>🌍 WorldLens</h1>
          <span className="tagline">
            {status.mode === "signbridge"
              ? "Sign Language Translation"
              : "Environmental Awareness"}
          </span>
        </div>
        <StatusBar status={status} />
      </header>

      {/* Main content */}
      <main className="app-main">
        {!session ? (
          /* Landing / connect screen */
          <div className="landing">
            <div className="landing-card">
              <h2>Welcome to WorldLens</h2>
              <p>
                An advanced dual-mode assistive vision platform powered by the
                Vision Agents SDK, Gemini 2.5 Flash Realtime, and GetStream Edge.
              </p>

              <div className="mode-indicator">
                <span className="mode-label">Current Mode:</span>
                <span className="mode-value">
                  {status.mode === "signbridge"
                    ? "🤟 SignBridge — Sign Language Translation"
                    : "👁️ GuideLens — Environmental Awareness"}
                </span>
              </div>

              <div className="status-check">
                <span
                  className={`status-dot ${
                    status.backendHealthy ? "healthy" : "unhealthy"
                  }`}
                />
                <span>
                  Backend:{" "}
                  {status.backendHealthy ? "Connected" : "Not reachable"}
                </span>
              </div>

              {error && <div className="error-banner">{error}</div>}

              <button
                className="btn btn-primary btn-large"
                onClick={handleStart}
                disabled={loading}
              >
                {loading ? "Connecting…" : "Start Session"}
              </button>

              <p className="hint">
                This will open your camera and microphone to connect with the
                WorldLens AI agent.
              </p>
            </div>
          </div>
        ) : (
          /* Active session */
          <div className="session-layout">
            {/* Video area */}
            <div className="video-area">
              <VideoRoom
                callId={callId}
                onLeave={handleStop}
              />
            </div>

            {/* Sidebar */}
            <aside className="sidebar">
              <ChatLog entries={transcript} />
              <TelemetryPanel data={telemetry} />
            </aside>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <span>WorldLens v0.1 • Vision Agents SDK • Gemini 2.5 Flash • GetStream Edge</span>
      </footer>
    </div>
  );
}

export default App;
