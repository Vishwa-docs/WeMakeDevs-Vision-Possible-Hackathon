/**
 * WorldLens — Main Application
 * Day 3: Core layout with Stream Video, status bar, chat log, mode toggle,
 * 3D Avatar with lip-sync, and OCR text overlay.
 */
import React, { useState, useCallback, useEffect, useRef } from "react";
import { useAgentSession } from "./hooks/useAgentSession";
import { StatusBar } from "./components/StatusBar";
import { VideoRoom } from "./components/VideoRoom";
import { ChatLog } from "./components/ChatLog";
import { TelemetryPanel } from "./components/TelemetryPanel";
import { AlertOverlay } from "./components/AlertOverlay";
import { ProviderSelector } from "./components/ProviderSelector";
import { Avatar3D } from "./components/Avatar3D";
import { OCROverlay } from "./components/OCROverlay";
import { ToastContainer, useToasts } from "./components/Toast";
import { getTranscript, clearTranscript, getTelemetry, pollHazardAlerts } from "./utils/api";
import type { TranscriptEntry, TelemetryData, HazardAlert } from "./types";
import type { FallbackEvent } from "./utils/api";
import "./App.css";

// ---------------------------------------------------------------------------
// Session-level Error Boundary — prevents the session view from blank-page
// ---------------------------------------------------------------------------
interface SEBProps {
  children: React.ReactNode;
  onReset?: () => void;
}
interface SEBState {
  error: Error | null;
}

class SessionErrorBoundary extends React.Component<SEBProps, SEBState> {
  constructor(props: SEBProps) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("[WorldLens][SessionErrorBoundary]", error, info.componentStack);
  }
  render() {
    if (this.state.error) {
      return (
        <div
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            padding: "2rem",
            color: "#e0e0e0",
          }}
        >
          <h2>⚠️ Session Error</h2>
          <pre
            style={{
              background: "#1a1a2e",
              color: "#ff6b6b",
              padding: "1rem",
              borderRadius: "8px",
              maxWidth: "600px",
              overflow: "auto",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
              fontSize: "0.8rem",
              marginBottom: "1rem",
            }}
          >
            {this.state.error.message}
          </pre>
          <button
            className="btn btn-primary"
            onClick={() => {
              this.setState({ error: null });
              this.props.onReset?.();
            }}
          >
            End Session &amp; Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

function App() {
  const {
    session,
    status,
    loading,
    error,
    startSession,
    stopSession,
    toggleMode,
  } = useAgentSession();
  const [modeMessage, setModeMessage] = useState<string | null>(null);
  const { toasts, addToast, dismissToast } = useToasts();

  // Handle provider fallback events → show toast
  const handleFallbackToast = useCallback(
    (event: FallbackEvent) => {
      addToast(
        `Provider ${event.original} failed (${event.reason}). Switched to ${event.fallback}.`,
        "warning",
        6000
      );
    },
    [addToast]
  );

  const [callId, setCallId] = useState(`worldlens-${Date.now()}`);
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const lastTranscriptTs = React.useRef(0);

  // Day 5: Hazard alert state (replaces old placeholder)
  const [alertActive, setAlertActive] = useState(false);
  const [currentAlert, setCurrentAlert] = useState<HazardAlert | null>(null);
  const lastHazardTs = useRef(0);
  const alertDeactivateTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Day 5: Real telemetry from backend
  const [telemetry, setTelemetry] = useState<TelemetryData>({
    mode: status.mode,
    uptime_seconds: 0,
    processors: [],
    processor_count: 0,
    aggregate: {
      total_frames_processed: 0,
      avg_inference_ms: 0,
      total_objects_detected: 0,
      total_hazards_detected: 0,
      total_gestures_detected: 0,
      total_ocr_calls: 0,
    },
    providers: { preferred: "gemini", chain: [], stats: {} },
    memory: { total_detections: 0, unique_objects: 0, recent_5min: 0 },
    navigation: { mode: "idle", scene_summary: "", pending_hazards: 0 },
  });

  const [startTime] = useState(Date.now());
  const [uptime, setUptime] = useState(0);
  const [agentSpeaking, setAgentSpeaking] = useState(false);
  const agentSpeechTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const onWindowError = (event: ErrorEvent) => {
      console.error("[WorldLens][window.error]", {
        message: event.message,
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
        error: event.error,
      });
    };

    const onUnhandledRejection = (event: PromiseRejectionEvent) => {
      console.error("[WorldLens][unhandledrejection]", event.reason);
    };

    window.addEventListener("error", onWindowError);
    window.addEventListener("unhandledrejection", onUnhandledRejection);

    return () => {
      window.removeEventListener("error", onWindowError);
      window.removeEventListener("unhandledrejection", onUnhandledRejection);
    };
  }, []);

  // Update uptime counter
  useEffect(() => {
    if (!session) return;
    const interval = setInterval(() => {
      setUptime((Date.now() - startTime) / 1000);
    }, 1000);
    return () => clearInterval(interval);
  }, [session, startTime]);

  // Poll backend for transcript entries while session is active
  useEffect(() => {
    if (!session) return;
    const interval = setInterval(async () => {
      const data = await getTranscript(lastTranscriptTs.current);
      if (data.entries.length > 0) {
        setTranscript((prev) => [
          ...prev,
          ...data.entries.map((e) => ({
            speaker: e.speaker as "user" | "agent",
            text: e.text,
            timestamp: e.timestamp,
          })),
        ]);
        lastTranscriptTs.current = data.entries[data.entries.length - 1].timestamp;

        // Detect agent speech for avatar lip-sync
        const hasAgentSpeech = data.entries.some((e) => e.speaker === "agent");
        if (hasAgentSpeech) {
          setAgentSpeaking(true);
          // Clear previous timer
          if (agentSpeechTimer.current) {
            clearTimeout(agentSpeechTimer.current);
          }
          // Stop speaking after 3s of no new agent transcript
          agentSpeechTimer.current = setTimeout(() => {
            setAgentSpeaking(false);
          }, 3000);
        }
      }
    }, 1500);
    return () => {
      clearInterval(interval);
      if (agentSpeechTimer.current) clearTimeout(agentSpeechTimer.current);
    };
  }, [session]);

  // Day 5: Poll backend for hazard alerts while session is active
  useEffect(() => {
    if (!session) return;
    const interval = setInterval(async () => {
      const data = await pollHazardAlerts(lastHazardTs.current);
      if (data.alerts && data.alerts.length > 0) {
        // Show the highest-priority (first) alert
        const top = data.alerts[0];
        setCurrentAlert({
          type: top.type || "hazard",
          text: top.text || top.type || "Obstacle detected",
          severity: top.severity || "warning",
          direction: top.direction || "center",
          sound: top.sound,
          duration_ms: top.duration_ms,
          priority: top.priority,
          timestamp: top.timestamp || Date.now(),
          distance: top.distance,
          growth_rate: top.growth_rate,
        });
        setAlertActive(true);
        lastHazardTs.current = Date.now();

        // Auto-deactivate after alert duration so next alert can trigger
        const dur = top.duration_ms || 3000;
        if (alertDeactivateTimer.current) clearTimeout(alertDeactivateTimer.current);
        alertDeactivateTimer.current = setTimeout(() => setAlertActive(false), dur + 200);
      }
    }, 2000);
    return () => {
      clearInterval(interval);
      if (alertDeactivateTimer.current) clearTimeout(alertDeactivateTimer.current);
    };
  }, [session]);

  // Day 5: Poll backend for real telemetry while session is active
  useEffect(() => {
    if (!session) return;
    const interval = setInterval(async () => {
      const data = await getTelemetry();
      if (data) {
        setTelemetry(data);
      }
    }, 3000);
    // Fetch immediately on mount
    getTelemetry().then((d) => d && setTelemetry(d));
    return () => clearInterval(interval);
  }, [session]);

  // Handle mode toggle
  const handleToggleMode = useCallback(async () => {
    const result = await toggleMode();
    if (result) {
      const msg = session
        ? `Switched to ${result.mode}. Takes effect next session.`
        : `Switched to ${result.mode}.`;
      setModeMessage(msg);
      setTimeout(() => setModeMessage(null), 3000);
    }
  }, [toggleMode, session]);

  // Telemetry uptime fallback (if backend /telemetry not yet responding)
  useEffect(() => {
    if (telemetry.uptime_seconds > 0) return; // real data available
    setTelemetry((prev) => ({ ...prev, uptime_seconds: uptime }));
  }, [uptime, telemetry.uptime_seconds]);

  const handleStart = useCallback(async () => {
    const id = `worldlens-${Date.now()}`;
    console.log("[WorldLens][App] Start Session clicked", { requestedCallId: id });
    setCallId(id);
    const started = await startSession(id);
    console.log("[WorldLens][App] startSession returned", started);
    if (started?.call_id) {
      setCallId(started.call_id);
      console.log("[WorldLens][App] callId set from backend session", {
        callId: started.call_id,
        sessionId: started.session_id,
      });
    } else {
      console.warn("[WorldLens][App] startSession returned null/invalid payload");
    }
  }, [startSession]);

  const handleStop = useCallback(async () => {
    console.log("[WorldLens][App] Stop Session requested", {
      sessionId: session?.session_id,
      callId: session?.call_id,
    });
    await stopSession();
    await clearTranscript();
    setTranscript([]);
    lastTranscriptTs.current = 0;
    console.log("[WorldLens][App] Session stopped and transcript cleared");
  }, [session?.call_id, session?.session_id, stopSession]);

  return (
    <div className="app">
      {/* Alert overlay for hazard warnings (Day 5: real hazard alerts) */}
      <AlertOverlay
        active={alertActive}
        alert={currentAlert}
        message={currentAlert?.text || "Obstacle detected!"}
        direction={currentAlert?.direction}
      />
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />

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
        <StatusBar
          status={status}
          onToggleMode={handleToggleMode}
          sessionActive={!!session}
        />
        <ProviderSelector onFallbackToast={handleFallbackToast} />
      </header>

      {/* Mode switch notification */}
      {modeMessage && (
        <div className="mode-notification">{modeMessage}</div>
      )}

      {/* Main content */}
      <main className="app-main">
        {!session ? (
          /* Landing / connect screen */
          <div className="landing">
            <div className="landing-card">
              <div className="landing-hero-icon">🌍</div>
              <h2>WorldLens</h2>
              <p className="landing-subtitle">
                AI-Powered Assistive Vision Platform
              </p>
              <p className="landing-desc">
                Dual-mode assistive vision powered by Vision Agents SDK,
                Gemini 2.5 Flash Realtime, and GetStream Edge.
              </p>

              {/* Mode cards */}
              <div className="mode-cards">
                <div className={`mode-card ${status.mode === "guidelens" ? "active" : ""}`}>
                  <span className="mode-card-icon">👁️</span>
                  <span className="mode-card-title">GuideLens</span>
                  <span className="mode-card-desc">Environmental awareness, hazard detection, navigation</span>
                </div>
                <div className={`mode-card ${status.mode === "signbridge" ? "active" : ""}`}>
                  <span className="mode-card-icon">🤟</span>
                  <span className="mode-card-title">SignBridge</span>
                  <span className="mode-card-desc">Sign language recognition, 3D avatar, translation</span>
                </div>
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
                {loading ? (
                  <>
                    <span className="btn-spinner" /> Connecting…
                  </>
                ) : (
                  "Start Session"
                )}
              </button>

              <p className="hint">
                Opens camera &amp; microphone to connect with WorldLens AI.
              </p>

              {/* Feature badges */}
              <div className="feature-badges">
                <span className="feature-badge">🧠 Gemini 2.5 Flash</span>
                <span className="feature-badge">📹 Real-time Video</span>
                <span className="feature-badge">🗣️ Voice AI</span>
                <span className="feature-badge">🗺️ Google Maps</span>
                <span className="feature-badge">🔍 YOLO Detection</span>
                <span className="feature-badge">✋ MediaPipe</span>
              </div>
            </div>
          </div>
        ) : (
          /* Active session */
          <SessionErrorBoundary onReset={handleStop}>
          <div className="session-layout">
            {/* Video area with OCR overlay */}
            <div className="video-area">
              <VideoRoom
                callId={session.call_id || callId}
                onLeave={handleStop}
              />
              <OCROverlay active={!!session} pollInterval={5000} />
            </div>

            {/* Sidebar */}
            <aside className="sidebar">
              {/* 3D Avatar (SignBridge mode) */}
              {status.mode === "signbridge" && (
                <div className="avatar-panel sidebar-section">
                  <div className="avatar-panel-header">
                    <span>🤟 SignBridge Avatar</span>
                    <span
                      className={`speaking-indicator ${
                        agentSpeaking ? "active" : "silent"
                      }`}
                    >
                      <span className="speaking-dot" />
                      {agentSpeaking ? "Speaking" : "Silent"}
                    </span>
                  </div>
                  <Avatar3D
                    isSpeaking={agentSpeaking}
                    style={{ height: 220 }}
                  />
                </div>
              )}
              <div className="sidebar-section">
                <div className="sidebar-section-header">
                  <span>💬 Conversation</span>
                  <span className="sidebar-section-badge">{transcript.length}</span>
                </div>
                <ChatLog entries={transcript} />
              </div>
              <div className="sidebar-section">
                <div className="sidebar-section-header">
                  <span>📊 Telemetry</span>
                  <span className="telemetry-live-indicator">
                    <span className="live-dot" /> LIVE
                  </span>
                </div>
                <TelemetryPanel data={telemetry} />
              </div>
            </aside>
          </div>
          </SessionErrorBoundary>
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
