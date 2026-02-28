/**
 * WorldLens — Telemetry / Metrics Panel (Day 5: Enterprise Edition)
 * Displays real-time metrics from the backend /telemetry endpoint.
 */
import type { TelemetryData } from "../types";

interface TelemetryPanelProps {
  data: TelemetryData;
}

export function TelemetryPanel({ data }: TelemetryPanelProps) {
  const agg = data.aggregate;
  const hasReal = agg !== undefined; // true once real data arrives

  return (
    <div className="telemetry-panel">
      <h3>
        Telemetry <span className="telemetry-live-dot" />
      </h3>
      <div className="telemetry-grid">
        {/* ─── Core ─── */}
        <div className="telemetry-section-label">Core</div>

        <div className="telemetry-item">
          <span className="telemetry-label">Mode</span>
          <span className="telemetry-value">
            {data.mode || data.activeVLM || "—"}
          </span>
        </div>

        <div className="telemetry-item">
          <span className="telemetry-label">Uptime</span>
          <span className="telemetry-value">
            {formatUptime(data.uptime_seconds ?? data.uptime ?? 0)}
          </span>
        </div>

        <div className="telemetry-item">
          <span className="telemetry-label">Processors</span>
          <span className="telemetry-value">
            {data.processor_count ?? data.processorCount ?? 0}
          </span>
        </div>

        {/* ─── Inference ─── */}
        {hasReal && (
          <>
            <div className="telemetry-section-label">Inference</div>

            <div className="telemetry-item">
              <span className="telemetry-label">Total Frames</span>
              <span className="telemetry-value neutral">
                {(agg.total_frames_processed ?? 0).toLocaleString()}
              </span>
            </div>

            <div className="telemetry-item">
              <span className="telemetry-label">Avg Latency</span>
              <span
                className={`telemetry-value ${latencyColor(
                  agg.avg_inference_ms ?? 0
                )}`}
              >
                {(agg.avg_inference_ms ?? 0).toFixed(1)}ms
              </span>
            </div>

            <div className="telemetry-item highlight">
              <span className="telemetry-label">Objects Detected</span>
              <span className="telemetry-value good">
                {(agg.total_objects_detected ?? 0).toLocaleString()}
              </span>
            </div>

            <div className="telemetry-item warn">
              <span className="telemetry-label">Hazards Detected</span>
              <span
                className={`telemetry-value ${
                  (agg.total_hazards_detected ?? 0) > 0 ? "warn" : "good"
                }`}
              >
                {agg.total_hazards_detected ?? 0}
              </span>
            </div>

            <div className="telemetry-item">
              <span className="telemetry-label">Gestures</span>
              <span className="telemetry-value neutral">
                {agg.total_gestures_detected ?? 0}
              </span>
            </div>

            <div className="telemetry-item">
              <span className="telemetry-label">OCR Calls</span>
              <span className="telemetry-value neutral">
                {agg.total_ocr_calls ?? 0}
              </span>
            </div>
          </>
        )}

        {/* ─── Providers ─── */}
        {data.providers && (
          <>
            <div className="telemetry-section-label">Providers</div>
            <div className="telemetry-item">
              <span className="telemetry-label">Preferred</span>
              <span className="telemetry-value">{data.providers.preferred}</span>
            </div>
            <div className="telemetry-item">
              <span className="telemetry-label">Chain</span>
              <span className="telemetry-value" style={{ fontSize: "0.65rem" }}>
                {data.providers.chain?.join(" → ") || "—"}
              </span>
            </div>
          </>
        )}

        {/* ─── Memory ─── */}
        {data.memory && (
          <>
            <div className="telemetry-section-label">Spatial Memory</div>
            <div className="telemetry-item">
              <span className="telemetry-label">Detections</span>
              <span className="telemetry-value neutral">
                {data.memory.total_detections}
              </span>
            </div>
            <div className="telemetry-item">
              <span className="telemetry-label">Unique Objects</span>
              <span className="telemetry-value good">
                {data.memory.unique_objects}
              </span>
            </div>
            <div className="telemetry-item">
              <span className="telemetry-label">Recent (5 min)</span>
              <span className="telemetry-value">
                {data.memory.recent_5min}
              </span>
            </div>
          </>
        )}

        {/* ─── Navigation ─── */}
        {data.navigation && (
          <>
            <div className="telemetry-section-label">Navigation</div>
            <div className="telemetry-item">
              <span className="telemetry-label">Nav Mode</span>
              <span className="telemetry-value">
                {data.navigation.mode}
              </span>
            </div>
            <div className="telemetry-item">
              <span className="telemetry-label">Pending Hazards</span>
              <span
                className={`telemetry-value ${
                  data.navigation.pending_hazards > 0 ? "warn" : "good"
                }`}
              >
                {data.navigation.pending_hazards}
              </span>
            </div>
          </>
        )}

        {/* ─── Per-processor details ─── */}
        {data.processors && data.processors.length > 0 && (
          <>
            <div className="telemetry-section-label">Processor Details</div>
            {data.processors.map((p, i) => (
              <div className="telemetry-item" key={(p as any).processor || p.name || i}>
                <span className="telemetry-label">{(p as any).processor || p.name || `proc-${i}`}</span>
                <span className="telemetry-value" style={{ fontSize: "0.65rem" }}>
                  {p.frames_processed ?? 0}f · {(p.avg_inference_ms ?? 0).toFixed(0)}ms
                </span>
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}

function formatUptime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}m ${s}s`;
}

function latencyColor(ms: number | undefined | null): string {
  if (!ms || ms < 30) return "good";
  if (ms < 80) return "neutral";
  if (ms < 150) return "warn";
  return "bad";
}
