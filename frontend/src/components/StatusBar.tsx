/**
 * WorldLens — Status indicator bar with mode toggle
 */
import type { AgentStatus } from "../types";

interface StatusBarProps {
  status: AgentStatus;
  onToggleMode?: () => void;
  sessionActive?: boolean;
}

export function StatusBar({ status, onToggleMode, sessionActive }: StatusBarProps) {
  return (
    <div className="status-bar">
      <div className="status-item">
        <span
          className={`status-dot ${status.backendHealthy ? "healthy" : "unhealthy"}`}
        />
        <span>Backend</span>
      </div>
      <div className="status-item">
        <span
          className={`status-dot ${status.connected ? "healthy" : "idle"}`}
        />
        <span>{status.connected ? "Connected" : "Idle"}</span>
      </div>
      <button
        className="mode-toggle-btn"
        onClick={onToggleMode}
        disabled={!status.backendHealthy}
        title={
          sessionActive
            ? "Mode will change on next session"
            : "Toggle between SignBridge and GuideLens"
        }
      >
        <span className="mode-toggle-icon">
          {status.mode === "signbridge" ? "🤟" : "👁️"}
        </span>
        <span className="mode-toggle-label">
          {status.mode === "signbridge" ? "SignBridge" : "GuideLens"}
        </span>
        <span className="mode-toggle-switch">⇄</span>
      </button>
    </div>
  );
}
