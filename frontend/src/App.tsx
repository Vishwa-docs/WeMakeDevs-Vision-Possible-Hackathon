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
import { getTranscript, clearTranscript } from "./utils/api";
import type { TranscriptEntry, TelemetryData } from "./types";
import type { FallbackEvent } from "./utils/api";
import "./App.css";

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
  const alertActive = false; // placeholder — will become useState once hazard events are wired up
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

  // Telemetry (Day 5: real metrics from backend)
  const telemetry: TelemetryData = {
    edgeLatency: 24,
    activeVLM:
      status.mode === "signbridge"
        ? "YOLO-Pose + Gemini + OCR"
        : "YOLO-Detect + Gemini + OCR",
    fps: status.mode === "signbridge" ? 10 : 5,
    processorCount: status.connected ? 2 : 0, // YOLO + OCR
    uptime,
  };

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
      {/* Alert overlay for hazard warnings */}
      <AlertOverlay active={alertActive} message="Obstacle approaching!" />
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
                    <span>SignBridge Avatar</span>
                    <span
                      className={`speaking-indicator ${
                        agentSpeaking ? "active" : "silent"
                      }`}
                    >
                      {agentSpeaking ? "Speaking..." : "Silent"}
                    </span>
                  </div>
                  <Avatar3D
                    isSpeaking={agentSpeaking}
                    style={{ height: 220 }}
                  />
                </div>
              )}
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
