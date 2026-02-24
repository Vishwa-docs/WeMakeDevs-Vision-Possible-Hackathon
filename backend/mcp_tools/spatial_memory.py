"""
Spatial Memory — SQLite-backed object/event memory
====================================================
Logs detected objects with timestamps for later recall.
The agent calls search_memory when a user asks "What did I see earlier?"

Day 4: Full implementation with async SQLite.
Day 1: Interface scaffold with in-memory fallback.
"""

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger("mcp.spatial_memory")


@dataclass
class MemoryEntry:
    """A single spatial memory record."""
    object_name: str
    timestamp: float
    location_hint: str = ""  # e.g., "left of frame", "center"
    confidence: float = 0.0
    frame_number: int = 0


class SpatialMemory:
    """
    In-memory spatial memory store.
    Day 4: Migrate to async SQLite (aiosqlite).
    """

    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self._entries: list[MemoryEntry] = []

    def log_detection(
        self,
        object_name: str,
        confidence: float = 0.0,
        location_hint: str = "",
        frame_number: int = 0,
    ) -> None:
        """Log a detected object into memory."""
        entry = MemoryEntry(
            object_name=object_name,
            timestamp=time.time(),
            location_hint=location_hint,
            confidence=confidence,
            frame_number=frame_number,
        )
        self._entries.append(entry)

        # Evict oldest
        while len(self._entries) > self.max_entries:
            self._entries.pop(0)

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """
        Search memory for objects matching the query.

        Args:
            query: Object name or keyword to search for.
            limit: Maximum results to return.

        Returns:
            List of matching memory entries as dicts.
        """
        query_lower = query.lower()
        matches = [
            {
                "object": e.object_name,
                "timestamp": e.timestamp,
                "time_ago": self._format_time_ago(e.timestamp),
                "location": e.location_hint,
                "confidence": e.confidence,
            }
            for e in reversed(self._entries)
            if query_lower in e.object_name.lower()
        ]
        return matches[:limit]

    def get_recent(self, n: int = 20) -> list[dict]:
        """Get the N most recent memory entries."""
        return [
            {
                "object": e.object_name,
                "timestamp": e.timestamp,
                "time_ago": self._format_time_ago(e.timestamp),
                "location": e.location_hint,
            }
            for e in reversed(self._entries[-n:])
        ]

    def _format_time_ago(self, ts: float) -> str:
        """Format a timestamp as a human-readable 'time ago' string."""
        diff = time.time() - ts
        if diff < 60:
            return f"{int(diff)}s ago"
        elif diff < 3600:
            return f"{int(diff / 60)}m ago"
        elif diff < 86400:
            return f"{int(diff / 3600)}h ago"
        return f"{int(diff / 86400)}d ago"

    @property
    def count(self) -> int:
        return len(self._entries)

    def clear(self):
        self._entries.clear()


# Module-level singleton for easy import
spatial_memory = SpatialMemory()
