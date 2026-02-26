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

export { STREAM_API_KEY, BACKEND_URL };
