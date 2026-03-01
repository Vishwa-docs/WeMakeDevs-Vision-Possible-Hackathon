/**
 * ProviderSelector — Lets user choose preferred VLM provider and see status.
 */
import { useState, useEffect, useCallback } from "react";
import {
  getProviders,
  setPreferredProvider,
  getFallbackEvents,
  type ProvidersStatus,
  type FallbackEvent,
} from "../utils/api";

const PROVIDER_META: Record<
  string,
  { label: string; icon: string; description: string }
> = {
  gemini: {
    label: "Gemini",
    icon: "🔮",
    description: "Google Gemini 2.5 Flash — fast, multimodal",
  },
  grok: {
    label: "Grok",
    icon: "🧠",
    description: "xAI Grok 4 — reasoning, vision",
  },
  azure_openai: {
    label: "Azure OpenAI",
    icon: "☁️",
    description: "GPT-4o via Azure — enterprise reliable",
  },
  nvidia: {
    label: "NVIDIA",
    icon: "🟢",
    description: "Cosmos Reason 2 via NGC — edge-optimised",
  },
  huggingface: {
    label: "HuggingFace",
    icon: "🤗",
    description: "Open models via Inference API",
  },
};

interface Props {
  onFallbackToast: (event: FallbackEvent) => void;
}

export function ProviderSelector({ onFallbackToast }: Props) {
  const [status, setStatus] = useState<ProvidersStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState(false);

  // Fetch provider status
  const fetchStatus = useCallback(async () => {
    try {
      const s = await getProviders();
      setStatus(s);
    } catch {
      // backend not available
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 15000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  // Poll for fallback events
  useEffect(() => {
    const poll = setInterval(async () => {
      try {
        const { events } = await getFallbackEvents();
        events.forEach(onFallbackToast);
      } catch {
        // ignore
      }
    }, 3000);
    return () => clearInterval(poll);
  }, [onFallbackToast]);

  const handleSelect = async (providerId: string) => {
    setLoading(true);
    try {
      await setPreferredProvider(providerId);
      // Re-fetch full status (includes health) instead of using the POST
      // response, which omits the health field
      await fetchStatus();
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  };

  if (!status) return null;

  const preferred = status.preferred;
  const preferredMeta = PROVIDER_META[preferred] || {
    label: preferred,
    icon: "⚙️",
    description: "",
  };

  return (
    <div className="provider-selector">
      {/* Collapsed: show current provider */}
      <button
        className="provider-toggle-btn"
        onClick={() => setExpanded(!expanded)}
        title="Select VLM provider"
      >
        <span className="provider-icon">{preferredMeta.icon}</span>
        <span className="provider-label">{preferredMeta.label}</span>
        <span className="chevron">{expanded ? "▲" : "▼"}</span>
      </button>

      {/* Expanded: provider list */}
      {expanded && (
        <div className="provider-dropdown">
          <div className="provider-dropdown-header">VLM Provider</div>
          {status.fallback_chain.map((pid) => {
            const meta = PROVIDER_META[pid] || {
              label: pid,
              icon: "⚙️",
              description: "",
            };
            const info = status.providers[pid];
            const healthy = status.health?.[pid] ?? false;
            const isPreferred = pid === preferred;
            const inCooldown = info?.in_cooldown ?? false;

            return (
              <button
                key={pid}
                className={`provider-option ${isPreferred ? "active" : ""} ${
                  !healthy ? "unavailable" : ""
                } ${inCooldown ? "cooldown" : ""}`}
                onClick={() => handleSelect(pid)}
                disabled={loading}
              >
                <span className="provider-option-icon">{meta.icon}</span>
                <div className="provider-option-info">
                  <span className="provider-option-name">
                    {meta.label}
                    {isPreferred && (
                      <span className="preferred-badge">preferred</span>
                    )}
                  </span>
                  <span className="provider-option-desc">{meta.description}</span>
                </div>
                <span
                  className={`provider-status-dot ${
                    inCooldown
                      ? "cooldown"
                      : healthy
                      ? "ready"
                      : "unavailable"
                  }`}
                  title={
                    inCooldown
                      ? `Cooldown: ${info?.last_error || "rate limited"}`
                      : healthy
                      ? "Available"
                      : "Not configured"
                  }
                />
              </button>
            );
          })}
          <div className="provider-dropdown-footer">
            Auto-fallback to next provider on failure
          </div>
        </div>
      )}
    </div>
  );
}
