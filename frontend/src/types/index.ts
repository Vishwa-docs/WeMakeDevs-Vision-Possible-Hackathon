/**
 * WorldLens — Type definitions
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

export interface HazardAlert {
  type: string;
  direction: "left" | "center" | "right";
  distance: "near" | "medium" | "far";
  confidence: number;
}

export interface TelemetryData {
  edgeLatency: number;
  activeVLM: string;
  fps: number;
  processorCount: number;
  uptime: number;
}
