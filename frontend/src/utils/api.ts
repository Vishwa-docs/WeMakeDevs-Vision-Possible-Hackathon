/**
 * WorldLens — Stream Video API client utilities
 */

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";
const STREAM_API_KEY = import.meta.env.VITE_STREAM_API_KEY || "";

/** Create a new agent session via the backend API */
export async function createSession(
  callType: string = "default",
  callId: string = `worldlens-${Date.now()}`
): Promise<{ session_id: string; call_type: string; call_id: string }> {
  const res = await fetch(`${BACKEND_URL}/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ call_type: callType, call_id: callId }),
  });
  if (!res.ok) throw new Error(`Failed to create session: ${res.statusText}`);
  return res.json();
}

/** End an agent session */
export async function endSession(sessionId: string): Promise<void> {
  await fetch(`${BACKEND_URL}/sessions/${sessionId}`, {
    method: "DELETE",
  });
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

export { STREAM_API_KEY, BACKEND_URL };
