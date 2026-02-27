/**
 * WorldLens — Stream Video API client utilities
 */

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";
const STREAM_API_KEY = import.meta.env.VITE_STREAM_API_KEY || "";

export async function getStreamConfig(): Promise<{ api_key: string }> {
  console.log("[WorldLens][api] GET /stream-config");
  const res = await fetch(`${BACKEND_URL}/stream-config`);
  if (!res.ok) throw new Error(`Failed to load stream config: ${res.statusText}`);
  const data = await res.json();
  console.log("[WorldLens][api] GET /stream-config success", {
    hasApiKey: Boolean(data?.api_key),
  });
  return data;
}

/** Create a new agent session via the backend API */
export async function createSession(
  callType: string = "default",
  callId: string = `worldlens-${Date.now()}`
): Promise<{ session_id: string; call_type: string; call_id: string }> {
  console.log("[WorldLens][api] POST /sessions", { callType, callId });
  const res = await fetch(`${BACKEND_URL}/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ call_type: callType, call_id: callId }),
  });
  if (!res.ok) throw new Error(`Failed to create session: ${res.statusText}`);
  const data = await res.json();
  console.log("[WorldLens][api] POST /sessions success", data);
  // SDK response has session_id + call_id + session_started_at; normalise
  return {
    session_id: data.session_id,
    call_type: callType,
    call_id: data.call_id,
  };
}

/** End an agent session */
export async function endSession(sessionId: string): Promise<void> {
  try {
    await fetch(`${BACKEND_URL}/sessions/${sessionId}`, {
      method: "DELETE",
    });
  } catch {
    // Best-effort — 404 is normal if agent already finished
  }
}

/** Get backend mode (signbridge / guidelens) */
export async function getAgentMode(): Promise<{ mode: string }> {
  const res = await fetch(`${BACKEND_URL}/mode`);
  if (!res.ok) return { mode: "unknown" };
  return res.json();
}

/** Toggle agent mode (signbridge ↔ guidelens) */
export async function switchMode(): Promise<{ mode: string; message: string }> {
  const res = await fetch(`${BACKEND_URL}/switch-mode`, { method: "POST" });
  if (!res.ok) throw new Error(`Failed to switch mode: ${res.statusText}`);
  return res.json();
}

/** Set a specific agent mode */
export async function setMode(
  mode: "signbridge" | "guidelens"
): Promise<{ mode: string }> {
  const res = await fetch(`${BACKEND_URL}/set-mode/${mode}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(`Failed to set mode: ${res.statusText}`);
  return res.json();
}

/** Check backend health */
export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BACKEND_URL}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Provider management
// ---------------------------------------------------------------------------

export interface ProviderInfo {
  provider: string;
  available: boolean;
  in_cooldown: boolean;
  last_error: string | null;
  total_calls: number;
  total_errors: number;
}

export interface ProvidersStatus {
  preferred: string;
  providers: Record<string, ProviderInfo>;
  fallback_chain: string[];
  health: Record<string, boolean>;
}

export interface FallbackEvent {
  original: string;
  fallback: string;
  reason: string;
  timestamp: number;
}

/** Get status of all VLM providers */
export async function getProviders(): Promise<ProvidersStatus> {
  const res = await fetch(`${BACKEND_URL}/providers`);
  if (!res.ok) throw new Error("Failed to fetch providers");
  return res.json();
}

/** Set the preferred VLM provider */
export async function setPreferredProvider(
  providerId: string
): Promise<ProvidersStatus> {
  const res = await fetch(
    `${BACKEND_URL}/providers/preferred/${providerId}`,
    { method: "POST" }
  );
  if (!res.ok) throw new Error("Failed to set preferred provider");
  return res.json();
}

/** Poll for fallback events (for toast notifications) */
export async function getFallbackEvents(): Promise<{
  events: FallbackEvent[];
}> {
  const res = await fetch(`${BACKEND_URL}/providers/fallback-events`);
  if (!res.ok) return { events: [] };
  return res.json();
}

/** Get transcript entries (optionally since a timestamp in ms) */
export async function getTranscript(
  since: number = 0
): Promise<{ entries: { speaker: string; text: string; timestamp: number }[] }> {
  try {
    const res = await fetch(
      `${BACKEND_URL}/transcript${since > 0 ? `?since=${since}` : ""}`
    );
    if (!res.ok) return { entries: [] };
    return res.json();
  } catch {
    return { entries: [] };
  }
}

/** Clear transcript log on session end */
export async function clearTranscript(): Promise<void> {
  try {
    await fetch(`${BACKEND_URL}/transcript`, { method: "DELETE" });
  } catch {
    // best-effort
  }
}

// ---------------------------------------------------------------------------
// OCR / VLM endpoints (Day 3)
// ---------------------------------------------------------------------------

export interface OCRResult {
  text: string;
  provider: string;
  timestamp: number;
  source?: string;
  frame_number?: number;
  error?: string;
}

export interface SceneDescription {
  description: string;
  provider: string;
  timestamp: number;
  error?: string;
}

/** Get cached OCR results for the overlay */
export async function getOCRResults(
  since: number = 0,
  limit: number = 10
): Promise<{ results: OCRResult[]; available: boolean }> {
  try {
    const res = await fetch(
      `${BACKEND_URL}/ocr-results?since=${since}&limit=${limit}`
    );
    if (!res.ok) return { results: [], available: false };
    return res.json();
  } catch {
    return { results: [], available: false };
  }
}

/** Manually trigger OCR read of the current frame */
export async function triggerOCRRead(
  prompt: string = ""
): Promise<OCRResult> {
  const res = await fetch(`${BACKEND_URL}/ocr/read`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  if (!res.ok) throw new Error("OCR read failed");
  return res.json();
}

/** Manually trigger a detailed scene description */
export async function triggerSceneDescription(
  prompt: string = ""
): Promise<SceneDescription> {
  const res = await fetch(`${BACKEND_URL}/ocr/describe`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  if (!res.ok) throw new Error("Scene description failed");
  return res.json();
}

export { STREAM_API_KEY, BACKEND_URL };
