/**
 * OCROverlay — Displays detected text from the backend OCR processor
 * ===================================================================
 * Day 3: Polls /ocr-results and renders text detections as a floating
 * overlay on top of the video feed. Results fade out after a timeout.
 */
import { useState, useEffect, useRef } from "react";
import { getOCRResults, type OCRResult } from "../utils/api";

interface OCROverlayProps {
  /** Whether to actively poll for results */
  active: boolean;
  /** Poll interval in ms (default: 5000) */
  pollInterval?: number;
  /** How long a result remains visible in ms (default: 15000) */
  displayDuration?: number;
}

export function OCROverlay({
  active,
  pollInterval = 5000,
  displayDuration = 15000,
}: OCROverlayProps) {
  const [results, setResults] = useState<OCRResult[]>([]);
  const lastTimestamp = useRef(0);

  // Poll OCR results from backend
  useEffect(() => {
    if (!active) return;

    const poll = async () => {
      try {
        const data = await getOCRResults(lastTimestamp.current);
        if (data.results && data.results.length > 0) {
          let shouldUpdateLastTimestamp = false;
          setResults((prev) => {
            const newResults = data.results.filter(
              (r: OCRResult) =>
                !prev.some((p) => Math.abs(p.timestamp - r.timestamp) < 0.5)
            );
            if (newResults.length === 0) return prev;
            shouldUpdateLastTimestamp = true;
            return [...prev, ...newResults].slice(-10); // Keep max 10
          });
          if (shouldUpdateLastTimestamp) {
            lastTimestamp.current =
              data.results[data.results.length - 1].timestamp;
          }
        }
      } catch {
        // Silent fail — OCR is non-critical
      }
    };

    const interval = setInterval(poll, pollInterval);
    poll(); // Initial poll
    return () => clearInterval(interval);
  }, [active, pollInterval]);

  // Expire old results
  useEffect(() => {
    if (results.length === 0) return;
    const timer = setInterval(() => {
      const cutoff = Date.now() / 1000 - displayDuration / 1000;
      setResults((prev) => prev.filter((r) => r.timestamp > cutoff));
    }, 2000);
    return () => clearInterval(timer);
  }, [results.length, displayDuration]);

  if (!active || results.length === 0) return null;

  return (
    <div className="ocr-overlay">
      <div className="ocr-overlay-header">
        <span className="ocr-icon">📝</span>
        <span>Detected Text</span>
      </div>
      <div className="ocr-results-list">
        {results.map((r, i) => {
          const age = Date.now() / 1000 - r.timestamp;
          const opacity = Math.max(0.3, 1 - age / (displayDuration / 1000));
          return (
            <div
              key={`${r.timestamp}-${i}`}
              className="ocr-result-item"
              style={{ opacity }}
            >
              <p className="ocr-text">{r.text}</p>
              <span className="ocr-meta">
                via {r.provider} • {formatAge(age)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function formatAge(seconds: number): string {
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${Math.round(seconds)}s ago`;
  return `${Math.round(seconds / 60)}m ago`;
}


