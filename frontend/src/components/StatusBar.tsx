/**
 * WorldLens — Status indicator bar
 */
import { AgentStatus } from "../types";

interface StatusBarProps {
  status: AgentStatus;
}

export function StatusBar({ status }: StatusBarProps) {
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
      <div className="status-item mode-badge">
        {status.mode === "signbridge" ? "🤟 SignBridge" : "👁️ GuideLens"}
      </div>
    </div>
  );
}
