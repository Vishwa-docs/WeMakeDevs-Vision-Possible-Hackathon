/**
 * WorldLens — Type definitions (Day 5: expanded for real telemetry & alerts)
 */

export type AgentMode = "signbridge" | "guidelens";

export interface SessionInfo {
  session_id: string;
  call_type: string;
  call_id: string;
}

export interface AgentStatus {
  connected: boolean;
  mode: AgentMode;
  backendHealthy: boolean;
}

export interface TranscriptEntry {
  speaker: "user" | "agent";
  text: string;
  timestamp: number;
}

export interface DetectionResult {
  objects: string[];
  timestamp: number;
  hazards: HazardAlert[];
}

// ---------------------------------------------------------------------------
// Hazard alerts (matches backend haptic-alert schema)
// ---------------------------------------------------------------------------

export interface HazardAlert {
  type: string;
  text: string;
  severity: "critical" | "warning" | "caution";
  direction: "left" | "center" | "right";
  sound?: "siren" | "beep" | "chime";
  duration_ms?: number;
  priority?: number;
  timestamp?: number;
  distance?: string;
  growth_rate?: number;
  class?: string;
  confidence?: number;
}

// ---------------------------------------------------------------------------
// Telemetry (matches backend /telemetry response)
// ---------------------------------------------------------------------------

/** Per-processor telemetry from the backend */
export interface ProcessorTelemetry {
  name?: string;
  processor?: string; // backend sends "processor" not "name"
  frames_processed?: number;
  avg_inference_ms?: number;
  total_inference_time?: number;
  uptime_seconds?: number;
  // GuideLens-specific
  total_objects_detected?: number;
  total_hazards_detected?: number;
  // SignBridge-specific
  total_gestures_detected?: number;
  total_persons_detected?: number;
  // OCR-specific
  total_ocr_calls?: number;
  total_scene_calls?: number;
}

/** Aggregate telemetry across all processors */
export interface AggregateTelemetry {
  total_frames_processed: number;
  avg_inference_ms: number;
  total_objects_detected: number;
  total_hazards_detected: number;
  total_gestures_detected: number;
  total_ocr_calls: number;
}

/** Full telemetry payload from GET /telemetry */
export interface TelemetryData {
  mode: string;
  uptime_seconds: number;
  processors: ProcessorTelemetry[];
  processor_count: number;
  aggregate: AggregateTelemetry;
  providers: {
    preferred: string;
    chain: string[];
    stats: Record<string, { calls: number; errors: number }>;
  };
  memory: {
    total_detections: number;
    unique_objects: number;
    recent_5min: number;
  };
  navigation: {
    mode: string;
    scene_summary: string;
    pending_hazards: number;
  };

  // ---- Convenience fields used by the panel (computed on frontend) ----
  /** Alias fields — filled by the polling hook for backward compat */
  edgeLatency?: number;
  activeVLM?: string;
  fps?: number;
  processorCount?: number;
  uptime?: number;
}
