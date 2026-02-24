/**
 * WorldLens — Transcript / Chat history log
 */
import { useEffect, useRef } from "react";
import type { TranscriptEntry } from "../types";

interface ChatLogProps {
  entries: TranscriptEntry[];
}

export function ChatLog({ entries }: ChatLogProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [entries]);

  if (entries.length === 0) {
    return (
      <div className="chat-log empty">
        <p>Conversation transcript will appear here…</p>
      </div>
    );
  }

  return (
    <div className="chat-log" ref={scrollRef}>
      {entries.map((entry, i) => (
        <div key={i} className={`chat-entry ${entry.speaker}`}>
          <span className="chat-speaker">
            {entry.speaker === "user" ? "You" : "WorldLens"}
          </span>
          <span className="chat-text">{entry.text}</span>
          <span className="chat-time">
            {new Date(entry.timestamp).toLocaleTimeString()}
          </span>
        </div>
      ))}
    </div>
  );
}
