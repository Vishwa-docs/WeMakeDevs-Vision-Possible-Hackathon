"""
Local Storage & Processing Utilities
======================================
Provides local frame storage, detection caching, and session management
that runs entirely on-device without external dependencies.
"""

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("utils.local_storage")


class LocalFrameStore:
    """
    Stores processed frames and detection results locally on disk.
    Useful for:
      - Offline analysis of captured sessions
      - Building training datasets from real-world detections
      - Session replay and debugging
    """

    def __init__(self, base_dir: str = "./data/frames", max_disk_mb: float = 500.0):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_disk_bytes = int(max_disk_mb * 1024 * 1024)
        self._session_id = f"session_{int(time.time())}"
        self._session_dir = self.base_dir / self._session_id
        self._session_dir.mkdir(exist_ok=True)
        self._frame_index = 0
        self._metadata: list[dict] = []
        logger.info("LocalFrameStore initialized: %s", self._session_dir)

    def save_frame(
        self,
        frame: np.ndarray,
        detections: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> str:
        """
        Save a frame and its associated detection data.

        Args:
            frame: BGR numpy array from OpenCV.
            detections: List of detection dicts (bboxes, labels, etc.).
            metadata: Additional metadata to store.

        Returns:
            Path to the saved frame file.
        """
        self._frame_index += 1
        ts = time.time()

        # Save frame as JPEG
        fname = f"frame_{self._frame_index:06d}.jpg"
        fpath = self._session_dir / fname
        cv2.imwrite(str(fpath), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Save metadata
        entry = {
            "frame_index": self._frame_index,
            "timestamp": ts,
            "filename": fname,
            "detections": detections or [],
            **(metadata or {}),
        }
        self._metadata.append(entry)

        # Periodic metadata flush
        if self._frame_index % 100 == 0:
            self._flush_metadata()
            self._check_disk_usage()

        return str(fpath)

    def _flush_metadata(self):
        """Write metadata index to disk."""
        meta_path = self._session_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "session_id": self._session_id,
                    "frame_count": self._frame_index,
                    "entries": self._metadata,
                },
                f,
                indent=2,
            )

    def _check_disk_usage(self):
        """Evict old sessions if total disk usage exceeds limit."""
        total = sum(
            f.stat().st_size
            for f in self.base_dir.rglob("*")
            if f.is_file()
        )
        if total > self.max_disk_bytes:
            # Remove oldest sessions
            sessions = sorted(
                [d for d in self.base_dir.iterdir() if d.is_dir()],
                key=lambda d: d.stat().st_mtime,
            )
            while total > self.max_disk_bytes * 0.8 and len(sessions) > 1:
                oldest = sessions.pop(0)
                if oldest != self._session_dir:
                    size = sum(f.stat().st_size for f in oldest.rglob("*"))
                    shutil.rmtree(oldest)
                    total -= size
                    logger.info("Evicted old session: %s (freed %.1f MB)", oldest.name, size / 1024 / 1024)

    def close(self):
        """Flush remaining metadata and finalize session."""
        self._flush_metadata()
        logger.info(
            "Session %s closed: %d frames saved to %s",
            self._session_id,
            self._frame_index,
            self._session_dir,
        )

    @property
    def session_dir(self) -> Path:
        return self._session_dir


class DetectionCache:
    """
    In-memory LRU cache for detection results to avoid redundant processing.
    Uses frame hashing to detect duplicate/similar frames.
    """

    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self._cache: dict[str, dict] = {}
        self._access_order: list[str] = []

    def _hash_frame(self, frame: np.ndarray) -> str:
        """Create a perceptual hash for quick frame similarity check."""
        small = cv2.resize(frame, (16, 16))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small
        mean = gray.mean()
        bits = (gray > mean).flatten()
        return "".join(str(int(b)) for b in bits)

    def get(self, frame: np.ndarray) -> dict | None:
        """Look up cached detections for a similar frame."""
        h = self._hash_frame(frame)
        if h in self._cache:
            # Move to end (most recent)
            self._access_order.remove(h)
            self._access_order.append(h)
            return self._cache[h]
        return None

    def put(self, frame: np.ndarray, result: dict) -> None:
        """Cache a detection result."""
        h = self._hash_frame(frame)
        self._cache[h] = result
        if h in self._access_order:
            self._access_order.remove(h)
        self._access_order.append(h)

        # Evict LRU
        while len(self._cache) > self.max_size:
            old_key = self._access_order.pop(0)
            self._cache.pop(old_key, None)

    def clear(self):
        self._cache.clear()
        self._access_order.clear()

    def __len__(self):
        return len(self._cache)


class SessionManager:
    """
    Manages agent session state and configuration persistence.
    Stores session data locally for crash recovery and analytics.
    """

    def __init__(self, data_dir: str = "./data/sessions"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._active_sessions: dict[str, dict] = {}

    def create_session(self, session_id: str, mode: str, call_id: str) -> dict:
        """Create and persist a new session."""
        session = {
            "session_id": session_id,
            "mode": mode,
            "call_id": call_id,
            "created_at": time.time(),
            "status": "active",
            "events": [],
        }
        self._active_sessions[session_id] = session
        self._persist(session_id)
        return session

    def log_event(self, session_id: str, event_type: str, data: Any = None) -> None:
        """Log an event to a session."""
        if session_id in self._active_sessions:
            self._active_sessions[session_id]["events"].append({
                "type": event_type,
                "timestamp": time.time(),
                "data": data,
            })

    def end_session(self, session_id: str) -> None:
        """Mark a session as ended and persist final state."""
        if session_id in self._active_sessions:
            self._active_sessions[session_id]["status"] = "ended"
            self._active_sessions[session_id]["ended_at"] = time.time()
            self._persist(session_id)
            del self._active_sessions[session_id]

    def _persist(self, session_id: str) -> None:
        """Write session data to disk."""
        if session_id in self._active_sessions:
            path = self.data_dir / f"{session_id}.json"
            with open(path, "w") as f:
                json.dump(self._active_sessions[session_id], f, indent=2, default=str)

    def get_session(self, session_id: str) -> dict | None:
        return self._active_sessions.get(session_id)

    @property
    def active_count(self) -> int:
        return len(self._active_sessions)
