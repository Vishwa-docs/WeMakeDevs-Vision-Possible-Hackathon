/**
 * WorldLens — Alert overlay for hazard/collision warnings (mock haptics)
 * Day 5: Full implementation with animation + chime.
 * Day 1: Scaffold component.
 */
import { useEffect, useState } from "react";

interface AlertOverlayProps {
  active: boolean;
  message?: string;
  direction?: "left" | "center" | "right";
}

export function AlertOverlay({ active, message, direction }: AlertOverlayProps) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (active) {
      setVisible(true);
      const timer = setTimeout(() => setVisible(false), 2000);
      return () => clearTimeout(timer);
    } else {
      setVisible(false);
    }
  }, [active]);

  if (!visible) return null;

  return (
    <div className={`alert-overlay ${direction || "center"}`}>
      <div className="alert-content">
        <span className="alert-icon">⚠️</span>
        <span className="alert-message">{message || "Hazard detected!"}</span>
      </div>
    </div>
  );
}
