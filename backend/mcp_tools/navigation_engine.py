"""
Navigation Engine — Smart Continuous Environmental Monitoring
==============================================================
Day 4: Implements the blind-navigation features inspired by
Blind-Nav and NVIDIA A-Eye for the Blind projects.

Three integrated sub-modes (all running simultaneously):
  1. Navigation Mode — continuous hazard monitoring, obstacle detection,
     road condition awareness (potholes, puddles, steps).
  2. Assistant Mode — on-demand Q&A about the environment, activated when
     the user asks a question. Only speaks when relevant.
  3. Reading Mode — text detection from signboards, books, phones.

Key design principles:
  - Only announces CHANGES in the environment (no repetitive announcements)
  - Priority-based alerts: safety first, then informational
  - Suppresses when user is speaking (don't interrupt)
  - On-demand assistant: stops continuous mode to answer questions
  - Cooldown windows per object to avoid spam
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Any

logger = logging.getLogger("worldlens.navigation")


# ---------------------------------------------------------------------------
# Announcement priority levels
# ---------------------------------------------------------------------------
class Priority:
    CRITICAL = 0   # Imminent collision, obstacle in path
    HIGH = 1       # Approaching hazard, vehicle nearby
    MEDIUM = 2     # New object detected, environment change
    LOW = 3        # Informational (distant objects, stable scene)
    SILENT = 4     # Suppressed (no change, repeated)


# ---------------------------------------------------------------------------
# Smart announcement tracker
# ---------------------------------------------------------------------------
@dataclass
class AnnouncementRecord:
    """Tracks when we last announced a specific thing."""
    content: str
    priority: int
    timestamp: float
    count: int = 1


class SmartAnnouncer:
    """
    Manages announcements to avoid repeating the same information.

    Rules:
      - Critical alerts: always announce immediately
      - New objects: announce once, then suppress for a cooldown
      - Environment changes: announce when significant change detected
      - Don't interrupt when user is speaking
    """

    # Cooldown per priority level (seconds)
    COOLDOWN = {
        Priority.CRITICAL: 1.5,    # Can repeat critical alerts quickly
        Priority.HIGH: 3.0,       # Approaching hazards — faster repeat
        Priority.MEDIUM: 5.0,     # New objects — announce frequently (3s cycle)
        Priority.LOW: 10.0,       # Informational — still proactive
        Priority.SILENT: float("inf"),
    }

    def __init__(self):
        self._history: dict[str, AnnouncementRecord] = {}
        self._user_speaking = False
        self._user_speaking_until = 0.0
        self._last_announcement_time = 0.0
        self._min_announcement_gap = 0.5  # min seconds between any announcements (proactive)
        self._queue: list[tuple[int, str, str]] = []  # (priority, key, text)

    def should_announce(self, key: str, priority: int) -> bool:
        """Check if we should announce this item given its history."""
        now = time.time()

        # Don't interrupt user
        if self._user_speaking or now < self._user_speaking_until:
            if priority > Priority.CRITICAL:  # Critical always announces
                return False

        # Check minimum gap between announcements
        if now - self._last_announcement_time < self._min_announcement_gap:
            if priority > Priority.CRITICAL:
                return False

        # Check cooldown for this specific item
        record = self._history.get(key)
        if record:
            cooldown = self.COOLDOWN.get(priority, 30.0)
            if now - record.timestamp < cooldown:
                return False

        return True

    def record_announcement(self, key: str, text: str, priority: int) -> None:
        """Record that we announced something."""
        now = time.time()
        existing = self._history.get(key)
        if existing:
            existing.timestamp = now
            existing.count += 1
            existing.content = text
            existing.priority = priority
        else:
            self._history[key] = AnnouncementRecord(
                content=text, priority=priority, timestamp=now
            )
        self._last_announcement_time = now

    def set_user_speaking(self, speaking: bool, duration: float = 2.0) -> None:
        """Mark user as speaking (suppress non-critical announcements)."""
        self._user_speaking = speaking
        if speaking:
            self._user_speaking_until = time.time() + duration

    def clear_old(self, max_age: float = 300.0) -> None:
        """Remove stale announcement records."""
        cutoff = time.time() - max_age
        to_remove = [
            k for k, v in self._history.items() if v.timestamp < cutoff
        ]
        for k in to_remove:
            del self._history[k]


# ---------------------------------------------------------------------------
# Environment state tracker (detects changes)
# ---------------------------------------------------------------------------
class EnvironmentState:
    """
    Tracks the current state of the environment to detect meaningful changes.

    Only triggers announcements when something NEW happens:
      - New object type appears
      - Object moves significantly closer
      - Object disappears (was near, now gone)
      - Text content changes
      - Scene layout changes significantly
    """

    def __init__(self):
        # Current scene objects: class_name -> {direction, distance, count, ts}
        self._current_objects: dict[str, dict] = {}
        self._previous_objects: dict[str, dict] = {}
        self._last_update_time = 0.0
        self._scene_hash = ""  # Simple hash of the scene composition

    def update(self, detections: list[dict]) -> list[dict]:
        """
        Update environment state with new detections.

        Returns a list of "change events" describing what changed.
        """
        self._previous_objects = self._current_objects.copy()
        now = time.time()
        self._last_update_time = now

        # Build current state
        new_objects: dict[str, dict] = {}
        for det in detections:
            cls = det.get("class", "")
            direction = det.get("direction", "")
            distance = det.get("distance", "")
            confidence = det.get("confidence", 0.0)

            key = cls
            if key in new_objects:
                # Keep the closest/highest confidence instance
                existing = new_objects[key]
                dist_order = {"near": 0, "medium": 1, "far": 2}
                if dist_order.get(distance, 2) < dist_order.get(existing.get("distance", "far"), 2):
                    new_objects[key] = {
                        "direction": direction,
                        "distance": distance,
                        "confidence": confidence,
                        "count": existing.get("count", 1) + 1,
                        "timestamp": now,
                    }
                else:
                    new_objects[key]["count"] = existing.get("count", 1) + 1
            else:
                new_objects[key] = {
                    "direction": direction,
                    "distance": distance,
                    "confidence": confidence,
                    "count": 1,
                    "timestamp": now,
                }

        self._current_objects = new_objects

        # Detect changes
        changes = []

        # New objects that weren't in previous state
        for cls, info in new_objects.items():
            prev = self._previous_objects.get(cls)
            if prev is None:
                changes.append({
                    "type": "new_object",
                    "class": cls,
                    "direction": info["direction"],
                    "distance": info["distance"],
                    "confidence": info["confidence"],
                    "priority": self._object_priority(cls, info["distance"]),
                })
            elif info["distance"] != prev.get("distance"):
                # Object moved closer or further
                dist_order = {"near": 0, "medium": 1, "far": 2}
                new_dist = dist_order.get(info["distance"], 2)
                old_dist = dist_order.get(prev.get("distance", "far"), 2)
                if new_dist < old_dist:
                    changes.append({
                        "type": "approaching",
                        "class": cls,
                        "direction": info["direction"],
                        "distance": info["distance"],
                        "previous_distance": prev.get("distance"),
                        "priority": Priority.HIGH if info["distance"] == "near" else Priority.MEDIUM,
                    })

        # Objects that disappeared (were near or medium)
        for cls, prev_info in self._previous_objects.items():
            if cls not in new_objects and prev_info.get("distance") in ("near", "medium"):
                changes.append({
                    "type": "cleared",
                    "class": cls,
                    "previous_distance": prev_info.get("distance"),
                    "priority": Priority.LOW,
                })

        return changes

    def _object_priority(self, cls: str, distance: str) -> int:
        """Determine announcement priority based on object type and distance."""
        # Vehicles and moving objects near the user are critical
        critical_types = {"car", "truck", "bus", "motorcycle", "bicycle"}
        hazard_types = {"person", "dog", "skateboard", "scooter"}

        if distance == "near":
            if cls in critical_types:
                return Priority.CRITICAL
            if cls in hazard_types:
                return Priority.HIGH
            return Priority.MEDIUM
        elif distance == "medium":
            if cls in critical_types:
                return Priority.HIGH
            return Priority.MEDIUM
        return Priority.LOW

    def get_current_summary(self) -> str:
        """Get a human-readable summary of the current environment."""
        if not self._current_objects:
            return "The path appears clear."

        parts = []
        # Sort by proximity (near first)
        dist_order = {"near": 0, "medium": 1, "far": 2}
        sorted_objects = sorted(
            self._current_objects.items(),
            key=lambda x: dist_order.get(x[1].get("distance", "far"), 2),
        )

        for cls, info in sorted_objects:
            count = info.get("count", 1)
            direction = info.get("direction", "")
            distance = info.get("distance", "")

            obj_str = f"{count} {cls}{'s' if count > 1 else ''}" if count > 1 else cls
            loc_parts = []
            if direction:
                loc_parts.append(f"to the {direction}")
            if distance:
                loc_parts.append(f"({distance})")
            if loc_parts:
                obj_str += " " + " ".join(loc_parts)
            parts.append(obj_str)

        return "I can see: " + ", ".join(parts) + "."


# ---------------------------------------------------------------------------
# Navigation Engine
# ---------------------------------------------------------------------------
class NavigationEngine:
    """
    Central engine that orchestrates all three sub-modes.

    Used by the GuideLens processor and main.py to:
      - Generate smart announcements from YOLO detections
      - Pause continuous announcements when user asks questions
      - Provide environment context for assistant Q&A
      - Build structured hazard alerts for the haptic UI
    """

    def __init__(self):
        self.announcer = SmartAnnouncer()
        self.environment = EnvironmentState()
        self._mode = "navigation"  # navigation | assistant | reading
        self._continuous = True
        self._assistant_active = False
        self._assistant_lock = asyncio.Lock()
        self._pending_announcements: list[dict] = []
        self._hazard_alerts: list[dict] = []

        # Active route tracking (for navigation status UI)
        self._active_route: Optional[dict] = None  # Set when directions are fetched
        self._navigation_active = False
        self._destination: str = ""
        self._route_steps: list[dict] = []
        self._current_step_index: int = 0

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        """Set the current sub-mode."""
        if mode in ("navigation", "assistant", "reading"):
            self._mode = mode
            logger.info("Navigation engine mode: %s", mode)

    def set_continuous(self, enabled: bool) -> None:
        """Enable/disable continuous announcements."""
        self._continuous = enabled

    def activate_assistant(self) -> None:
        """Temporarily pause navigation to handle a user question."""
        self._assistant_active = True
        self._mode = "assistant"

    def deactivate_assistant(self) -> None:
        """Resume navigation after answering a question."""
        self._assistant_active = False
        self._mode = "navigation"

    def process_detections(self, detections: list[dict]) -> list[dict]:
        """
        Process a batch of YOLO detections and return formatted announcements.

        Returns a list of announcement dicts:
          {"text": str, "priority": int, "type": str}
        """
        if not detections:
            return []

        # Don't generate navigation announcements during Q&A
        if self._assistant_active:
            return []

        # Update environment state and get changes
        changes = self.environment.update(detections)

        announcements = []
        for change in changes:
            key = f"{change['type']}:{change['class']}"
            priority = change.get("priority", Priority.MEDIUM)

            if not self.announcer.should_announce(key, priority):
                continue

            text = self._format_change(change)
            if text:
                announcements.append({
                    "text": text,
                    "priority": priority,
                    "type": change["type"],
                    "class": change["class"],
                })
                self.announcer.record_announcement(key, text, priority)

                # Track hazard alerts separately for the UI
                if priority <= Priority.HIGH:
                    self._hazard_alerts.append({
                        "text": text,
                        "priority": priority,
                        "type": change["type"],
                        "class": change["class"],
                        "timestamp": time.time(),
                    })

        # Trim hazard alerts (keep last 20)
        if len(self._hazard_alerts) > 20:
            self._hazard_alerts = self._hazard_alerts[-20:]

        return announcements

    def get_environment_summary(self) -> str:
        """Get current environment summary for assistant mode Q&A."""
        return self.environment.get_current_summary()

    def get_hazard_alerts(self, since: float = 0) -> list[dict]:
        """Get recent hazard alerts (for frontend haptic/visual feedback)."""
        if since > 0:
            return [a for a in self._hazard_alerts if a.get("timestamp", 0) > since]
        return list(self._hazard_alerts)

    def pop_hazard_alerts(self, since: float = 0) -> list[dict]:
        """Get and clear hazard alerts since timestamp (consumed by frontend)."""
        if since > 0:
            alerts = [a for a in self._hazard_alerts if a.get("timestamp", 0) > since]
        else:
            alerts = list(self._hazard_alerts)
        # Remove returned alerts
        if alerts:
            returned_ts = {a.get("timestamp", 0) for a in alerts}
            self._hazard_alerts = [
                a for a in self._hazard_alerts
                if a.get("timestamp", 0) not in returned_ts
            ]
        return alerts

    def clear_hazard_alerts(self) -> None:
        self._hazard_alerts.clear()

    def on_user_speech(self) -> None:
        """Called when user starts speaking — suppress non-critical announcements."""
        self.announcer.set_user_speaking(True, duration=2.0)

    def on_user_speech_end(self) -> None:
        """Called when user stops speaking — resume announcements."""
        self.announcer.set_user_speaking(False)

    # ------------------------------------------------------------------
    # Route tracking (for navigation status UI)
    # ------------------------------------------------------------------
    def set_active_route(self, destination: str, steps: list[dict],
                         total_distance: str = "", total_duration: str = "") -> None:
        """Set an active walking route (from get_walking_directions)."""
        self._navigation_active = True
        self._destination = destination
        self._route_steps = steps
        self._current_step_index = 0
        self._active_route = {
            "destination": destination,
            "total_distance": total_distance,
            "total_duration": total_duration,
            "step_count": len(steps),
        }
        logger.info("Navigation route set: %s (%s)", destination, total_distance)

    def clear_route(self) -> None:
        """Clear the active navigation route."""
        self._navigation_active = False
        self._destination = ""
        self._route_steps = []
        self._current_step_index = 0
        self._active_route = None
        logger.info("Navigation route cleared")

    def get_navigation_status(self) -> dict:
        """Get full navigation status for frontend UI."""
        return {
            "mode": self._mode,
            "navigation_active": self._navigation_active,
            "destination": self._destination,
            "current_step": self._current_step_index,
            "total_steps": len(self._route_steps),
            "active_route": self._active_route,
            "current_instruction": (
                self._route_steps[self._current_step_index]["instruction"]
                if self._route_steps and self._current_step_index < len(self._route_steps)
                else ""
            ),
            "scene_summary": self.environment.get_current_summary(),
            "pending_hazards": len(self._hazard_alerts),
            "continuous": self._continuous,
        }

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------
    def _format_change(self, change: dict) -> str:
        """Format a detected change into a spoken announcement."""
        change_type = change["type"]
        cls = change["class"]
        direction = change.get("direction", "")
        distance = change.get("distance", "")

        if change_type == "new_object":
            dir_str = f" to your {direction}" if direction else ""
            dist_str = f", {distance}" if distance else ""
            if distance == "near":
                return f"Caution: {cls}{dir_str}{dist_str}."
            return f"{cls.capitalize()} detected{dir_str}{dist_str}."

        elif change_type == "approaching":
            prev = change.get("previous_distance", "")
            dir_str = f" from the {direction}" if direction else ""
            if distance == "near":
                return f"Warning: {cls} approaching{dir_str}, now very close!"
            return f"{cls.capitalize()} getting closer{dir_str}."

        elif change_type == "cleared":
            return f"{cls.capitalize()} is no longer nearby."

        return ""


# Module-level singleton
navigation_engine = NavigationEngine()
