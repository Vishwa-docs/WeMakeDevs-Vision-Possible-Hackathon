/**
 * WorldLens — Custom hook for managing agent sessions
 */
import { useState, useCallback, useRef, useEffect } from "react";
import {
  createSession,
  endSession,
  getAgentMode,
  checkHealth,
  switchMode as apiSwitchMode,
} from "../utils/api";
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
      console.log("[WorldLens][useAgentSession] startSession begin", {
        requestedCallId: callId,
      });
      try {
        const s = await createSession("default", callId);
        console.log("[WorldLens][useAgentSession] createSession success", s);
        setSession(s);
        sessionRef.current = s;
        setStatus((prev) => ({ ...prev, connected: true }));
        return s;
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Failed to start session";
        console.error("[WorldLens][useAgentSession] createSession failed", err);
        setError(msg);
        return null;
      } finally {
        setLoading(false);
        console.log("[WorldLens][useAgentSession] startSession end");
      }
    },
    []
  );

  const stopSession = useCallback(async () => {
    if (sessionRef.current) {
      console.log("[WorldLens][useAgentSession] stopSession begin", {
        sessionId: sessionRef.current.session_id,
        callId: sessionRef.current.call_id,
      });
      try {
        await endSession(sessionRef.current.session_id);
        console.log("[WorldLens][useAgentSession] endSession success");
      } catch {
        // Best-effort cleanup — 404 is normal if agent already finished
        console.warn("[WorldLens][useAgentSession] endSession failed (ignored)");
      }
      sessionRef.current = null;
      setSession(null);
      setStatus((prev) => ({ ...prev, connected: false }));
      console.log("[WorldLens][useAgentSession] stopSession complete");
    }
  }, []);

  const toggleMode = useCallback(async () => {
    try {
      const result = await apiSwitchMode();
      setStatus((prev) => ({
        ...prev,
        mode: (result.mode as AgentMode) || prev.mode,
      }));
      return result;
    } catch (err) {
      const msg =
        err instanceof Error ? err.message : "Failed to switch mode";
      setError(msg);
      return null;
    }
  }, []);

  // Cleanup on unmount — use a small delay so React StrictMode's
  // unmount-then-remount cycle doesn't kill the session.
  useEffect(() => {
    return () => {
      const sid = sessionRef.current?.session_id;
      if (sid) {
        // Delay so if StrictMode re-mounts immediately the ref is
        // restored before this fires.  On a real unmount the timeout
        // runs and the session is cleaned up.
        setTimeout(() => {
          if (!sessionRef.current) {
            endSession(sid).catch(() => {});
          }
        }, 100);
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
    toggleMode,
  };
}
