/**
 * Toast — Ephemeral notification for provider fallback events etc.
 */
import { useEffect, useState, useCallback } from "react";

export interface ToastItem {
  id: string;
  message: string;
  type: "info" | "warning" | "error";
  duration?: number;
}

interface Props {
  toasts: ToastItem[];
  onDismiss: (id: string) => void;
}

export function ToastContainer({ toasts, onDismiss }: Props) {
  return (
    <div className="toast-container">
      {toasts.map((t) => (
        <Toast key={t.id} item={t} onDismiss={onDismiss} />
      ))}
    </div>
  );
}

function Toast({
  item,
  onDismiss,
}: {
  item: ToastItem;
  onDismiss: (id: string) => void;
}) {
  const [exiting, setExiting] = useState(false);

  useEffect(() => {
    const duration = item.duration ?? 5000;
    const timer = setTimeout(() => {
      setExiting(true);
      setTimeout(() => onDismiss(item.id), 300); // wait for exit animation
    }, duration);
    return () => clearTimeout(timer);
  }, [item, onDismiss]);

  return (
    <div className={`toast toast-${item.type} ${exiting ? "toast-exit" : ""}`}>
      <span className="toast-icon">
        {item.type === "warning" ? "⚠️" : item.type === "error" ? "❌" : "ℹ️"}
      </span>
      <span className="toast-message">{item.message}</span>
      <button
        className="toast-close"
        onClick={() => onDismiss(item.id)}
        aria-label="Dismiss"
      >
        ×
      </button>
    </div>
  );
}

/** Hook that provides toast state management. */
export function useToasts() {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const addToast = useCallback(
    (
      message: string,
      type: ToastItem["type"] = "info",
      duration?: number
    ) => {
      const id = `toast-${Date.now()}-${Math.random().toString(36).slice(2)}`;
      setToasts((prev) => [...prev, { id, message, type, duration }]);
      return id;
    },
    []
  );

  const dismissToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  return { toasts, addToast, dismissToast };
}
