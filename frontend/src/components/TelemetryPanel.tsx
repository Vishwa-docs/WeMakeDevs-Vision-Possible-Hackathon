/**
 * WorldLens — Telemetry / metrics panel
 */
import type { TelemetryData } from "../types";

interface TelemetryPanelProps {
  data: TelemetryData;
}

export function TelemetryPanel({ data }: TelemetryPanelProps) {
  return (
    <div className="telemetry-panel">
      <h3>Telemetry</h3>
      <div className="telemetry-grid">
        <div className="telemetry-item">
          <span className="telemetry-label">Edge Latency</span>
          <span className="telemetry-value">{data.edgeLatency}ms</span>
        </div>
        <div className="telemetry-item">
          <span className="telemetry-label">Active VLM</span>
          <span className="telemetry-value">{data.activeVLM}</span>
        </div>
        <div className="telemetry-item">
          <span className="telemetry-label">FPS</span>
          <span className="telemetry-value">{data.fps}</span>
        </div>
        <div className="telemetry-item">
          <span className="telemetry-label">Processors</span>
          <span className="telemetry-value">{data.processorCount}</span>
        </div>
        <div className="telemetry-item">
          <span className="telemetry-label">Uptime</span>
          <span className="telemetry-value">{formatUptime(data.uptime)}</span>
        </div>
      </div>
    </div>
  );
}

function formatUptime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}m ${s}s`;
}
