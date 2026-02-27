"""
Spatial Memory — Async SQLite-backed Object/Event Memory
=========================================================
Day 4: Full async SQLite implementation with deduplication,
time-ago formatting, and background detection logging.

Logs detected objects with timestamps, location hints, and confidence
scores for later recall. The agent calls ``search_memory`` when a user
asks "What did I see earlier?" or "Have you seen my keys?"

Schema:
    detections(
        id INTEGER PRIMARY KEY,
        object_name TEXT,
        timestamp REAL,
        location_hint TEXT,
        direction TEXT,
        distance TEXT,
        confidence REAL,
        frame_number INTEGER,
        session_id TEXT
    )
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import aiosqlite

logger = logging.getLogger("mcp.spatial_memory")

# Default DB path — same directory as this file
_DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "spatial_memory.db"
)


@dataclass
class MemoryEntry:
    """A single spatial memory record."""

    object_name: str
    timestamp: float
    location_hint: str = ""
    direction: str = ""
    distance: str = ""
    confidence: float = 0.0
    frame_number: int = 0
    session_id: str = ""
    id: int = 0


class SpatialMemory:
    """
    Async SQLite-backed spatial memory with deduplication.

    Features:
      - Async insert/query via aiosqlite
      - Dedup: won't re-log the same object+direction+distance within a cooldown
      - Time-ago formatted search results
      - Background batch-insert for high throughput
      - Session-aware logging
    """

    # Don't re-log the same object in the same position within this window
    DEDUP_COOLDOWN_SECONDS = 5.0

    def __init__(
        self,
        db_path: str = _DEFAULT_DB_PATH,
        max_entries: int = 10000,
        batch_interval: float = 2.0,
    ):
        self.db_path = db_path
        self.max_entries = max_entries
        self.batch_interval = batch_interval

        self._db: Optional[aiosqlite.Connection] = None
        self._initialised = False
        self._lock = asyncio.Lock()

        # Batch insert buffer
        self._pending: list[MemoryEntry] = []
        self._batch_task: Optional[asyncio.Task] = None
        self._running = False

        # Dedup tracker: key = (object_name, direction, distance) -> last_ts
        self._dedup_cache: dict[tuple, float] = {}

        # Session tracking
        self._session_id: str = ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def initialise(self) -> None:
        """Open DB and create tables if needed."""
        if self._initialised:
            return

        async with self._lock:
            if self._initialised:
                return

            db_dir = os.path.dirname(self.db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

            self._db = await aiosqlite.connect(self.db_path)
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute("PRAGMA synchronous=NORMAL")

            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    object_name TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    location_hint TEXT DEFAULT '',
                    direction TEXT DEFAULT '',
                    distance TEXT DEFAULT '',
                    confidence REAL DEFAULT 0.0,
                    frame_number INTEGER DEFAULT 0,
                    session_id TEXT DEFAULT ''
                )
            """)

            # Index for fast lookups
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_detections_name
                ON detections(object_name)
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_detections_timestamp
                ON detections(timestamp DESC)
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_detections_session
                ON detections(session_id, timestamp DESC)
            """)

            await self._db.commit()
            self._initialised = True
            self._running = True

            # Start background batch inserter
            self._batch_task = asyncio.create_task(self._batch_insert_loop())

            logger.info(
                "SpatialMemory initialised: %s (max=%d)",
                self.db_path,
                self.max_entries,
            )

    async def close(self) -> None:
        """Flush pending inserts and close DB."""
        self._running = False
        if self._batch_task and not self._batch_task.done():
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass

        # Flush remaining
        await self._flush_pending()

        if self._db:
            await self._db.close()
            self._db = None
            self._initialised = False
            logger.info("SpatialMemory closed")

    def set_session_id(self, session_id: str) -> None:
        self._session_id = session_id

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------
    async def log_detection(
        self,
        object_name: str,
        confidence: float = 0.0,
        location_hint: str = "",
        direction: str = "",
        distance: str = "",
        frame_number: int = 0,
    ) -> bool:
        """
        Log a detected object into memory.

        Returns True if the detection was logged, False if deduped.
        """
        if not self._initialised:
            await self.initialise()

        # Dedup check
        dedup_key = (object_name.lower(), direction, distance)
        now = time.time()
        last_ts = self._dedup_cache.get(dedup_key, 0.0)
        if now - last_ts < self.DEDUP_COOLDOWN_SECONDS:
            return False

        self._dedup_cache[dedup_key] = now

        entry = MemoryEntry(
            object_name=object_name,
            timestamp=now,
            location_hint=location_hint,
            direction=direction,
            distance=distance,
            confidence=confidence,
            frame_number=frame_number,
            session_id=self._session_id,
        )
        self._pending.append(entry)
        return True

    async def log_detection_batch(
        self, detections: list[dict]
    ) -> int:
        """
        Log multiple detections at once from the processor.

        Each dict should have keys: object_name/class, confidence, direction,
        distance, location_hint, frame_number.

        Returns the number of detections actually logged (after dedup).
        """
        logged = 0
        for det in detections:
            ok = await self.log_detection(
                object_name=det.get("class", det.get("object_name", "")),
                confidence=det.get("confidence", 0.0),
                direction=det.get("direction", ""),
                distance=det.get("distance", ""),
                location_hint=det.get("location_hint", ""),
                frame_number=det.get("frame_number", 0),
            )
            if ok:
                logged += 1
        return logged

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------
    async def search(self, query: str, limit: int = 10) -> list[dict]:
        """
        Search memory for objects matching the query.

        Supports partial matching and returns results sorted by recency.
        """
        if not self._initialised:
            await self.initialise()

        # Flush pending first so search sees latest data
        await self._flush_pending()

        query_lower = f"%{query.lower()}%"
        async with self._lock:
            cursor = await self._db.execute(
                """
                SELECT object_name, timestamp, location_hint, direction,
                       distance, confidence, frame_number, session_id
                FROM detections
                WHERE LOWER(object_name) LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (query_lower, limit),
            )
            rows = await cursor.fetchall()

        results = []
        for row in rows:
            results.append({
                "object": row[0],
                "timestamp": row[1],
                "time_ago": self._format_time_ago(row[1]),
                "location": row[2],
                "direction": row[3],
                "distance": row[4],
                "confidence": round(row[5], 2),
                "frame_number": row[6],
                "session_id": row[7],
            })
        return results

    async def get_recent(self, n: int = 20) -> list[dict]:
        """Get the N most recent memory entries."""
        if not self._initialised:
            await self.initialise()

        await self._flush_pending()

        async with self._lock:
            cursor = await self._db.execute(
                """
                SELECT object_name, timestamp, location_hint, direction,
                       distance, confidence
                FROM detections
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (n,),
            )
            rows = await cursor.fetchall()

        return [
            {
                "object": row[0],
                "timestamp": row[1],
                "time_ago": self._format_time_ago(row[1]),
                "location": row[2],
                "direction": row[3],
                "distance": row[4],
                "confidence": round(row[5], 2),
            }
            for row in rows
        ]

    async def get_summary(self) -> dict:
        """Get a high-level summary of spatial memory."""
        if not self._initialised:
            await self.initialise()

        await self._flush_pending()

        async with self._lock:
            # Total count
            cursor = await self._db.execute("SELECT COUNT(*) FROM detections")
            total = (await cursor.fetchone())[0]

            # Unique objects
            cursor = await self._db.execute(
                "SELECT COUNT(DISTINCT object_name) FROM detections"
            )
            unique = (await cursor.fetchone())[0]

            # Top objects by frequency
            cursor = await self._db.execute("""
                SELECT object_name, COUNT(*) as cnt
                FROM detections
                GROUP BY object_name
                ORDER BY cnt DESC
                LIMIT 10
            """)
            top_objects = [
                {"object": row[0], "count": row[1]}
                for row in await cursor.fetchall()
            ]

            # Last 5 minutes count
            five_min_ago = time.time() - 300
            cursor = await self._db.execute(
                "SELECT COUNT(*) FROM detections WHERE timestamp > ?",
                (five_min_ago,),
            )
            recent_count = (await cursor.fetchone())[0]

        return {
            "total_detections": total,
            "unique_objects": unique,
            "recent_5min": recent_count,
            "top_objects": top_objects,
            "session_id": self._session_id,
        }

    async def get_environment_context(self) -> str:
        """
        Get a natural-language summary of the recent environment.
        Used to provide context for the on-demand assistant mode.
        """
        recent = await self.get_recent(30)
        if not recent:
            return "No objects have been detected yet."

        # Group by object type
        groups: dict[str, list] = defaultdict(list)
        for r in recent:
            groups[r["object"]].append(r)

        parts = []
        for obj_name, entries in groups.items():
            latest = entries[0]
            count = len(entries)
            direction = latest.get("direction", "")
            distance = latest.get("distance", "")
            time_ago = latest.get("time_ago", "")

            location_str = ""
            if direction:
                location_str += f"to the {direction}"
            if distance:
                location_str += f" ({distance})"

            if count > 1:
                parts.append(
                    f"{obj_name} (seen {count} times, last {time_ago} {location_str})"
                )
            else:
                parts.append(f"{obj_name} ({time_ago} {location_str})")

        return "Recently detected: " + "; ".join(parts)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    async def evict_old(self) -> int:
        """Remove oldest entries when over max_entries."""
        if not self._db:
            return 0

        async with self._lock:
            cursor = await self._db.execute("SELECT COUNT(*) FROM detections")
            count = (await cursor.fetchone())[0]

            if count <= self.max_entries:
                return 0

            to_delete = count - self.max_entries
            await self._db.execute(
                """
                DELETE FROM detections
                WHERE id IN (
                    SELECT id FROM detections
                    ORDER BY timestamp ASC
                    LIMIT ?
                )
                """,
                (to_delete,),
            )
            await self._db.commit()
            logger.info("Evicted %d old detections", to_delete)
            return to_delete

    async def clear(self) -> None:
        """Clear all entries."""
        if not self._db:
            return
        async with self._lock:
            await self._db.execute("DELETE FROM detections")
            await self._db.commit()
        self._dedup_cache.clear()
        logger.info("SpatialMemory cleared")

    async def get_count(self) -> int:
        if not self._db:
            return 0
        async with self._lock:
            cursor = await self._db.execute("SELECT COUNT(*) FROM detections")
            return (await cursor.fetchone())[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _flush_pending(self) -> None:
        """Insert all pending entries into the DB."""
        if not self._pending or not self._db:
            return

        async with self._lock:
            entries = self._pending.copy()
            self._pending.clear()

            if not entries:
                return

            await self._db.executemany(
                """
                INSERT INTO detections
                    (object_name, timestamp, location_hint, direction,
                     distance, confidence, frame_number, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        e.object_name,
                        e.timestamp,
                        e.location_hint,
                        e.direction,
                        e.distance,
                        e.confidence,
                        e.frame_number,
                        e.session_id,
                    )
                    for e in entries
                ],
            )
            await self._db.commit()

            if len(entries) > 5:
                logger.debug("Flushed %d detections to SQLite", len(entries))

    async def _batch_insert_loop(self) -> None:
        """Background loop that flushes pending inserts periodically."""
        while self._running:
            await asyncio.sleep(self.batch_interval)
            if not self._running:
                break
            try:
                await self._flush_pending()
                # Periodic eviction check
                await self.evict_old()
            except Exception as e:
                logger.error("Batch insert error: %s", e)

    @staticmethod
    def _format_time_ago(ts: float) -> str:
        """Format a timestamp as a human-readable 'time ago' string."""
        diff = time.time() - ts
        if diff < 0:
            return "just now"
        if diff < 5:
            return "just now"
        if diff < 60:
            return f"{int(diff)}s ago"
        elif diff < 3600:
            minutes = int(diff / 60)
            return f"{minutes}m ago"
        elif diff < 86400:
            hours = int(diff / 3600)
            return f"{hours}h ago"
        days = int(diff / 86400)
        return f"{days}d ago"


# Module-level singleton for easy import
spatial_memory = SpatialMemory()
