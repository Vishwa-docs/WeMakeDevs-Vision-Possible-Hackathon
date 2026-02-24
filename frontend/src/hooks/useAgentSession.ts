/**
 * WorldLens — Custom hook for managing agent sessions
 */
import { useState, useCallback, useRef, useEffect } from "react";
import { createSession, endSession, getAgentMode, checkHealth } from "../utils/api";
import type { SessionInfo, AgentMode, AgentStatus } from "../types";

export function useAgentSession() {
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [status, setStatus] = useState<AgentStatus>({
    connected: false,
    mode: "guidelens",
    backendHealthy: false,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sessionRef = useRef<SessionInfo | null>(null);

  // Check backend health on mount
  useEffect(() => {
    const check = async () => {
      const healthy = await checkHealth();
      const modeRes = healthy ? await getAgentMode() : { mode: "unknown" };
      setStatus((prev) => ({
        ...prev,
        backendHealthy: healthy,
        mode: (modeRes.mode as AgentMode) || "guidelens",
      }));
    };
    check();
    const interval = setInterval(check, 10000);
    return () => clearInterval(interval);
  }, []);

  const startSession = useCallback(
    async (callId?: string) => {
      setLoading(true);
      setError(null);
      try {
        const s = await createSession("default", callId);
        setSession(s);
        sessionRef.current = s;
        setStatus((prev) => ({ ...prev, connected: true }));
        return s;
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Failed to start session";
        setError(msg);
        return null;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const stopSession = useCallback(async () => {
    if (sessionRef.current) {
      try {
        await endSession(sessionRef.current.session_id);
      } catch {
        // Best-effort cleanup
      }
      sessionRef.current = null;
      setSession(null);
      setStatus((prev) => ({ ...prev, connected: false }));
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (sessionRef.current) {
        endSession(sessionRef.current.session_id).catch(() => {});
      }
    };
  }, []);

  return {
    session,
    status,
    loading,
    error,
    startSession,
    stopSession,
  };
}
