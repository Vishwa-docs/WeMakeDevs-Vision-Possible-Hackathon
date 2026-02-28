/**
 * WorldLens V1.1 — Navigation Status Indicator
 * Shows whether navigation mode is active, current route info,
 * and real-time scene awareness status.
 */
import { useState, useEffect } from "react";

const BACKEND_URL =
  import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

interface NavStatus {
  mode: string;
  navigation_active: boolean;
  destination: string;
  current_step: number;
  total_steps: number;
  current_instruction: string;
  scene_summary: string;
  pending_hazards: number;
  continuous: boolean;
  active_route?: {
    destination: string;
    total_distance: string;
    total_duration: string;
    step_count: number;
  } | null;
}

interface NavigationStatusProps {
  active: boolean;
  pollInterval?: number;
}

export function NavigationStatus({
  active,
  pollInterval = 2000,
}: NavigationStatusProps) {
  const [status, setStatus] = useState<NavStatus | null>(null);

  useEffect(() => {
    if (!active) return;

    async function fetchStatus() {
      try {
        const res = await fetch(`${BACKEND_URL}/navigation/status`);
        if (res.ok) {
          const data = await res.json();
          setStatus(data);
        }
      } catch {
        // Ignore - backend might not be ready
      }
    }

    fetchStatus();
    const interval = setInterval(fetchStatus, pollInterval);
    return () => clearInterval(interval);
  }, [active, pollInterval]);

  if (!active || !status) return null;

  const modeLabels: Record<string, string> = {
    navigation: "Navigation",
    assistant: "Assistant",
    reading: "Reading",
  };

  const modeIcons: Record<string, string> = {
    navigation: "🧭",
    assistant: "💡",
    reading: "📖",
  };

  return (
    <div className="navigation-status">
      {/* Mode indicator */}
      <div className="nav-status-header">
        <span className="nav-mode-badge">
          <span className="nav-mode-icon">
            {modeIcons[status.mode] || "🧭"}
          </span>
          <span className="nav-mode-label">
            {modeLabels[status.mode] || status.mode}
          </span>
          {status.continuous && (
            <span className="nav-live-dot" title="Continuous monitoring active">
              ●
            </span>
          )}
        </span>

        {status.pending_hazards > 0 && (
          <span className="nav-hazard-badge" title="Active hazard alerts">
            ⚠️ {status.pending_hazards}
          </span>
        )}
      </div>

      {/* Active route */}
      {status.navigation_active && status.active_route && (
        <div className="nav-route-info">
          <div className="nav-route-destination">
            <span className="nav-route-icon">📍</span>
            <span className="nav-route-dest-text">
              {status.active_route.destination}
            </span>
          </div>
          <div className="nav-route-meta">
            <span>{status.active_route.total_distance}</span>
            <span className="nav-route-sep">•</span>
            <span>{status.active_route.total_duration}</span>
            <span className="nav-route-sep">•</span>
            <span>
              Step {status.current_step + 1}/{status.total_steps}
            </span>
          </div>
          {status.current_instruction && (
            <div className="nav-current-step">
              <span className="nav-step-arrow">→</span>
              <span>{status.current_instruction}</span>
            </div>
          )}
        </div>
      )}

      {/* Scene summary */}
      {status.scene_summary && status.scene_summary !== "The path appears clear." && (
        <div className="nav-scene-summary">
          <span className="nav-scene-icon">👁️</span>
          <span className="nav-scene-text">{status.scene_summary}</span>
        </div>
      )}
    </div>
  );
}
