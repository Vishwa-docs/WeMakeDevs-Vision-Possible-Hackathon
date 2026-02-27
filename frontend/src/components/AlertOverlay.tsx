/**
 * WorldLens — Alert Overlay for Hazard/Collision Warnings
 * Day 5: Full implementation with directional animation, severity-based
 * styling, Web Audio API alert chimes, and auto-dismiss.
 */
import { useEffect, useState, useRef, useCallback } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export interface HazardAlert {
  text: string;
  severity: "critical" | "warning" | "caution";
  direction: "left" | "center" | "right";
  sound?: "siren" | "beep" | "chime";
  duration_ms?: number;
  priority?: number;
  timestamp?: number;
  type?: string;
  class?: string;
  distance?: string;
  growth_rate?: number;
}

interface AlertOverlayProps {
  /** Triggers show when a new alert arrives (truthy = show, falsy = hide) */
  active: boolean;
  /** The current alert to display */
  alert?: HazardAlert | null;
  /** Fallback simple message */
  message?: string;
  /** Fallback direction */
  direction?: "left" | "center" | "right";
}

// ---------------------------------------------------------------------------
// Web Audio — alert sound generator
// ---------------------------------------------------------------------------
class AlertSoundPlayer {
  private ctx: AudioContext | null = null;
  private _lastPlayTime = 0;
  private _minGapMs = 800;

  private getContext(): AudioContext {
    if (!this.ctx || this.ctx.state === "closed") {
      this.ctx = new AudioContext();
    }
    if (this.ctx.state === "suspended") {
      this.ctx.resume();
    }
    return this.ctx;
  }

  play(type: "siren" | "beep" | "chime" = "beep") {
    const now = Date.now();
    if (now - this._lastPlayTime < this._minGapMs) return;
    this._lastPlayTime = now;

    try {
      const ctx = this.getContext();
      switch (type) {
        case "siren":
          this.playSiren(ctx);
          break;
        case "chime":
          this.playChime(ctx);
          break;
        case "beep":
        default:
          this.playBeep(ctx);
          break;
      }
    } catch (e) {
      console.warn("[AlertSound] Failed to play:", e);
    }
  }

  private playBeep(ctx: AudioContext) {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.type = "square";
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.3);
    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + 0.3);
  }

  private playSiren(ctx: AudioContext) {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.type = "sawtooth";
    // Wailing siren effect
    osc.frequency.setValueAtTime(600, ctx.currentTime);
    osc.frequency.linearRampToValueAtTime(1200, ctx.currentTime + 0.25);
    osc.frequency.linearRampToValueAtTime(600, ctx.currentTime + 0.5);
    osc.frequency.linearRampToValueAtTime(1200, ctx.currentTime + 0.75);
    gain.gain.setValueAtTime(0.12, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.8);
    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + 0.8);
  }

  private playChime(ctx: AudioContext) {
    // Two-tone gentle chime
    [523.25, 659.25].forEach((freq, i) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.type = "sine";
      osc.frequency.setValueAtTime(freq, ctx.currentTime + i * 0.15);
      gain.gain.setValueAtTime(0.1, ctx.currentTime + i * 0.15);
      gain.gain.exponentialRampToValueAtTime(
        0.001,
        ctx.currentTime + i * 0.15 + 0.4
      );
      osc.start(ctx.currentTime + i * 0.15);
      osc.stop(ctx.currentTime + i * 0.15 + 0.4);
    });
  }
}

const alertSound = new AlertSoundPlayer();

// ---------------------------------------------------------------------------
// Severity config
// ---------------------------------------------------------------------------
const SEVERITY_CONFIG = {
  critical: {
    borderColor: "#ef4444",
    bgColor: "rgba(239, 68, 68, 0.15)",
    glowColor: "#ef444488",
    icon: "🚨",
    label: "DANGER",
    pulseSpeed: "0.4s",
    defaultSound: "siren" as const,
  },
  warning: {
    borderColor: "#f59e0b",
    bgColor: "rgba(245, 158, 11, 0.12)",
    glowColor: "#f59e0b66",
    icon: "⚠️",
    label: "WARNING",
    pulseSpeed: "0.6s",
    defaultSound: "beep" as const,
  },
  caution: {
    borderColor: "#3b82f6",
    bgColor: "rgba(59, 130, 246, 0.1)",
    glowColor: "#3b82f644",
    icon: "ℹ️",
    label: "CAUTION",
    pulseSpeed: "1s",
    defaultSound: "chime" as const,
  },
};

const DIRECTION_ARROWS: Record<string, string> = {
  left: "⬅️",
  center: "⬆️",
  right: "➡️",
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function AlertOverlay({
  active,
  alert,
  message,
  direction,
}: AlertOverlayProps) {
  const [visible, setVisible] = useState(false);
  const [currentAlert, setCurrentAlert] = useState<HazardAlert | null>(null);
  const dismissTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const prevAlertRef = useRef<string>("");

  // Determine severity and direction from alert or props
  const severity = currentAlert?.severity || "warning";
  const dir = currentAlert?.direction || direction || "center";
  const text = currentAlert?.text || message || "Hazard detected!";
  const config = SEVERITY_CONFIG[severity];

  const dismiss = useCallback(() => {
    setVisible(false);
    if (dismissTimer.current) {
      clearTimeout(dismissTimer.current);
      dismissTimer.current = null;
    }
  }, []);

  useEffect(() => {
    if (active && (alert || message)) {
      // Prevent re-triggering same alert
      const alertKey = alert
        ? `${alert.text}:${alert.timestamp || 0}`
        : message || "";
      if (alertKey === prevAlertRef.current) return;
      prevAlertRef.current = alertKey;

      setCurrentAlert(alert || null);
      setVisible(true);

      // Play sound
      const soundType = alert?.sound || config.defaultSound;
      alertSound.play(soundType);

      // Haptic feedback (vibration API for mobile)
      if (navigator.vibrate) {
        const pattern =
          severity === "critical"
            ? [200, 100, 200, 100, 200]
            : severity === "warning"
            ? [200, 100, 200]
            : [100];
        navigator.vibrate(pattern);
      }

      // Auto-dismiss
      const duration =
        alert?.duration_ms || (severity === "critical" ? 3000 : 2000);
      if (dismissTimer.current) clearTimeout(dismissTimer.current);
      dismissTimer.current = setTimeout(dismiss, duration);
    } else if (!active) {
      dismiss();
    }

    return () => {
      if (dismissTimer.current) clearTimeout(dismissTimer.current);
    };
  }, [active, alert, message, severity, config.defaultSound, dismiss]);

  if (!visible) return null;

  return (
    <div
      className="alert-overlay-v2"
      style={
        {
          "--alert-border-color": config.borderColor,
          "--alert-bg-color": config.bgColor,
          "--alert-glow": config.glowColor,
          "--alert-pulse-speed": config.pulseSpeed,
        } as React.CSSProperties
      }
    >
      {/* Directional border glow */}
      <div className={`alert-border-glow alert-dir-${dir}`} />

      {/* Central alert card */}
      <div className={`alert-card alert-severity-${severity}`}>
        <div className="alert-card-inner">
          {/* Direction arrow */}
          <span className="alert-direction-arrow">
            {DIRECTION_ARROWS[dir] || "⬆️"}
          </span>

          {/* Icon + severity label */}
          <div className="alert-severity-badge">
            <span className="alert-icon-v2">{config.icon}</span>
            <span className="alert-severity-label">{config.label}</span>
          </div>

          {/* Message */}
          <p className="alert-message-v2">{text}</p>

          {/* Extra info */}
          {currentAlert?.distance && (
            <span className="alert-distance-tag">
              {currentAlert.distance}
              {currentAlert.growth_rate && currentAlert.growth_rate > 0
                ? " — approaching"
                : ""}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
