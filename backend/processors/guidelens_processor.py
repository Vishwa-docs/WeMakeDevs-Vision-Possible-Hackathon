"""
GuideLens Processor — Environmental Awareness for Visually Impaired
====================================================================
Day 4: Enhanced with spatial memory integration, navigation engine,
and smart continuous monitoring.

Architecture:
  Video Frame → YOLO Detection → Bounding Boxes → Hazard Analyser
                      ↓                   ↓             ↓
               Annotated Frame    NavigationEngine  SpatialMemory
                      ↓             ↓                    ↓
              Published Video   Smart Alerts        SQLite DB

Features:
  - Detects 80 COCO object classes via YOLOv11
  - Filters for hazard-relevant objects (person, car, bicycle, truck, etc.)
  - Tracks bounding-box growth rate to detect approaching objects
  - Estimates object direction (left / centre / right)
  - Estimates object distance (near / medium / far) from bbox area ratio
  - Emits structured events for hazards and general detections
  - Logs detections to async SQLite spatial memory
  - Smart announcements via NavigationEngine (change-only, priority-based)
  - Integrated Navigation, Assistant, and Reading sub-modes
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import av
import aiortc
import cv2
import numpy as np

from vision_agents.core.events import BaseEvent
from vision_agents.core.processors import VideoProcessorPublisher
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_track import QueuedVideoTrack

logger = logging.getLogger("guidelens.processor")


# ---------------------------------------------------------------------------
# Hazard configuration
# ---------------------------------------------------------------------------
# Objects that are safety-relevant for a visually impaired user
HAZARD_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "fire hydrant", "stop sign",
    "dog", "horse", "skateboard", "scooter",
}

# Colour palette for object classes (BGR for OpenCV)
CLASS_COLORS = {
    "person": (0, 200, 255),      # orange
    "bicycle": (255, 200, 0),     # cyan-ish
    "car": (0, 0, 255),           # red
    "motorcycle": (0, 0, 200),    # dark red
    "bus": (0, 100, 255),         # dark orange
    "truck": (0, 50, 200),        # maroon
    "traffic light": (0, 255, 0), # green
    "stop sign": (50, 50, 255),   # red
    "dog": (200, 150, 0),         # teal
}
DEFAULT_COLOR = (200, 200, 200)   # grey for other classes

# Proximity thresholds (bbox_area / frame_area)
DISTANCE_THRESHOLDS = {
    "near": 0.15,    # bbox > 15% of frame → near
    "medium": 0.05,  # bbox > 5% of frame → medium
}

# Approaching speed threshold (area-ratio growth per second)
APPROACH_RATE_THRESHOLD = 0.03  # 3 %/s growth → approaching


# ---------------------------------------------------------------------------
# Custom Events
# ---------------------------------------------------------------------------
@dataclass
class ObjectDetectedEvent(BaseEvent):
    """Emitted when objects are detected in the frame."""
    type: str = field(default="guidelens.object_detected", init=False)
    objects: list = field(default_factory=list)
    frame_number: int = 0
    timestamp_unix: float = 0.0


@dataclass
class HazardDetectedEvent(BaseEvent):
    """Emitted when a hazard is approaching or is dangerously close."""
    type: str = field(default="guidelens.hazard_detected", init=False)
    hazard_type: str = ""
    distance_estimate: str = ""   # "near", "medium", "far"
    direction: str = ""           # "left", "centre", "right"
    confidence: float = 0.0
    growth_rate: float = 0.0      # bbox area-ratio growth per second
    bbox_area_ratio: float = 0.0  # current bbox_area / frame_area


@dataclass
class SceneSummaryEvent(BaseEvent):
    """Periodic summary of all objects visible in the scene."""
    type: str = field(default="guidelens.scene_summary", init=False)
    summary: str = ""
    object_counts: dict = field(default_factory=dict)
    timestamp_unix: float = 0.0


# ---------------------------------------------------------------------------
# Bbox Tracker — per-class area history for proximity estimation
# ---------------------------------------------------------------------------
class BboxTracker:
    """
    Tracks the largest bounding-box area per object class across frames
    to estimate approach speed (growth rate).
    """

    def __init__(self, history_seconds: float = 3.0):
        self._history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=30)
        )
        self._history_seconds = history_seconds

    def update(self, class_name: str, area_ratio: float, timestamp: float):
        """Record a new area measurement for a class."""
        self._history[class_name].append((timestamp, area_ratio))

    def growth_rate(self, class_name: str) -> float:
        """
        Compute the area-ratio growth rate (per second) for a class.
        Positive = approaching, negative = receding.
        """
        history = self._history.get(class_name)
        if not history or len(history) < 2:
            return 0.0

        # Use oldest and newest entries within the history window
        t0, a0 = history[0]
        t1, a1 = history[-1]
        dt = t1 - t0
        if dt < 0.1:
            return 0.0
        return (a1 - a0) / dt

    def clear(self):
        self._history.clear()


# ---------------------------------------------------------------------------
# GuideLens Processor
# ---------------------------------------------------------------------------
class GuideLensProcessor(VideoProcessorPublisher):
    """
    Full YOLO object-detection processor for environmental-awareness.

    Receives video frames, detects objects, analyses hazard proximity,
    and publishes annotated frames with labelled bounding boxes.

    Day 4 enhancements:
      - Integrated with SpatialMemory (async SQLite) for detection logging
      - Integrated with NavigationEngine for smart change-only announcements
      - Background spatial memory sync loop
    """

    name = "guidelens_detection"

    def __init__(
        self,
        fps: int = 5,
        conf_threshold: float = 0.4,
        model_path: str = "yolo11n.pt",
        device: str = "cpu",
        scene_summary_interval: float = 10.0,
    ):
        self.fps = fps
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        self.device = device
        self.scene_summary_interval = scene_summary_interval

        self._forwarder: Optional[VideoForwarder] = None
        self._video_track = QueuedVideoTrack()
        self._frame_count = 0
        self._model = None
        self._bbox_tracker = BboxTracker()
        self._agent = None
        self._events = None
        self._processing = False  # inference mutex
        self._last_summary_time = 0.0
        self._detection_log: list[dict] = []

        # Day 4: Spatial memory & navigation engine integration
        self._spatial_memory = None
        self._navigation_engine = None

        self._load_model()

    def _load_model(self):
        """Load the YOLOv11 object-detection model."""
        try:
            from ultralytics import YOLO

            logger.info(
                "Loading YOLO detection model: %s (device: %s)",
                self.model_path,
                self.device,
            )
            self._model = YOLO(self.model_path)
            logger.info("YOLO detection model loaded successfully")
        except Exception as e:
            logger.error("Failed to load YOLO detection model: %s", e)
            logger.warning(
                "GuideLens will run in pass-through mode (no detection)"
            )

    # ------------------------------------------------------------------
    # Agent lifecycle
    # ------------------------------------------------------------------
    def set_spatial_memory(self, memory) -> None:
        """Inject the SpatialMemory instance for detection logging."""
        self._spatial_memory = memory
        logger.info("GuideLensProcessor: spatial memory connected")

    def set_navigation_engine(self, engine) -> None:
        """Inject the NavigationEngine for smart announcements."""
        self._navigation_engine = engine
        logger.info("GuideLensProcessor: navigation engine connected")

    def attach_agent(self, agent):
        """Register custom events with the agent's event system."""
        self._agent = agent
        self._events = agent.events
        self._events.register(ObjectDetectedEvent)
        self._events.register(HazardDetectedEvent)
        self._events.register(SceneSummaryEvent)
        logger.info("GuideLensProcessor attached to agent")

    async def process_video(
        self,
        track: aiortc.VideoStreamTrack,
        participant_id: Optional[str],
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        """Start processing incoming video frames."""
        if self._forwarder:
            await self._forwarder.remove_frame_handler(self._process_frame)

        self._forwarder = shared_forwarder
        self._forwarder.add_frame_handler(
            self._process_frame,
            fps=float(self.fps),
            name="guidelens_detection",
        )
        logger.info(
            "GuideLensProcessor started (fps=%d, conf=%.2f, model=%s)",
            self.fps,
            self.conf_threshold,
            self.model_path,
        )

    # ------------------------------------------------------------------
    # Frame processing pipeline
    # ------------------------------------------------------------------
    async def _process_frame(self, frame: av.VideoFrame) -> None:
        """Process a single video frame with YOLO object detection."""
        if self._processing:
            await self._video_track.add_frame(frame)
            return

        self._processing = True
        try:
            self._frame_count += 1
            timestamp = time.time()

            img = frame.to_ndarray(format="rgb24")
            h, w = img.shape[:2]
            frame_area = float(h * w)

            if self._model is not None:
                annotated_img, detections = await self._run_detection(
                    img, frame_area
                )

                if detections and self._events:
                    # --- Object detected event ---
                    object_names = [d["class"] for d in detections]
                    self._events.send(
                        ObjectDetectedEvent(
                            objects=object_names,
                            frame_number=self._frame_count,
                            timestamp_unix=timestamp,
                        )
                    )

                    # --- Hazard analysis ---
                    await self._analyse_hazards(
                        detections, w, frame_area, timestamp
                    )

                    # --- Scene summary (periodic) ---
                    if timestamp - self._last_summary_time > self.scene_summary_interval:
                        await self._emit_scene_summary(detections, timestamp)
                        self._last_summary_time = timestamp

                    # --- Log for spatial memory ---
                    self._log_detections(detections, timestamp)

                new_frame = av.VideoFrame.from_ndarray(
                    annotated_img, format="rgb24"
                )
            else:
                new_frame = frame

            await self._video_track.add_frame(new_frame)

            if self._frame_count % 100 == 0:
                logger.info(
                    "GuideLens: %d frames processed", self._frame_count
                )
        finally:
            self._processing = False

    async def _run_detection(
        self, img: np.ndarray, frame_area: float
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Run YOLO object detection in a thread-pool executor.
        Returns (annotated_image, list_of_detection_dicts).
        """
        loop = asyncio.get_event_loop()

        def _detect():
            results = self._model(
                img,
                conf=self.conf_threshold,
                verbose=False,
                device=self.device,
            )

            detections: list[dict] = []
            annotated = img.copy()

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = self._model.names.get(cls_id, f"class_{cls_id}")
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()

                    bbox_w = x2 - x1
                    bbox_h = y2 - y1
                    bbox_area = bbox_w * bbox_h
                    area_ratio = bbox_area / frame_area if frame_area > 0 else 0
                    cx = (x1 + x2) / 2.0

                    det = {
                        "class": cls_name,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                        "area_ratio": area_ratio,
                        "center_x": cx,
                        "center_y": (y1 + y2) / 2.0,
                        "is_hazard": cls_name in HAZARD_CLASSES,
                    }
                    detections.append(det)

                    # Draw bounding box
                    color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
                    ix1, iy1 = int(x1), int(y1)
                    ix2, iy2 = int(x2), int(y2)
                    thickness = 3 if det["is_hazard"] else 2
                    cv2.rectangle(
                        annotated, (ix1, iy1), (ix2, iy2), color, thickness
                    )

                    # Label
                    label = f"{cls_name} {conf:.0%}"
                    distance = self._estimate_distance(area_ratio)
                    label += f" [{distance}]"
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    cv2.rectangle(
                        annotated,
                        (ix1, iy1 - th - 8),
                        (ix1 + tw + 4, iy1),
                        color,
                        -1,
                    )
                    cv2.putText(
                        annotated,
                        label,
                        (ix1 + 2, iy1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

            # HUD overlay
            hazard_count = sum(1 for d in detections if d["is_hazard"])
            cv2.putText(
                annotated,
                f"GuideLens | Objects: {len(detections)} | Hazards: {hazard_count} | Frame: {self._frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            return annotated, detections

        return await loop.run_in_executor(None, _detect)

    # ------------------------------------------------------------------
    # Hazard analysis
    # ------------------------------------------------------------------
    async def _analyse_hazards(
        self,
        detections: list[dict],
        frame_width: int,
        frame_area: float,
        timestamp: float,
    ) -> None:
        """Analyse detections for approaching hazards."""
        for det in detections:
            if not det["is_hazard"]:
                continue

            cls = det["class"]
            area_ratio = det["area_ratio"]
            direction = self._estimate_direction(det["center_x"], frame_width)
            distance = self._estimate_distance(area_ratio)

            # Update tracker
            self._bbox_tracker.update(cls, area_ratio, timestamp)
            growth = self._bbox_tracker.growth_rate(cls)

            # Emit hazard event when object is near or approaching fast
            should_alert = (
                distance == "near"
                or (distance == "medium" and growth > APPROACH_RATE_THRESHOLD)
            )

            if should_alert and self._events:
                self._events.send(
                    HazardDetectedEvent(
                        hazard_type=cls,
                        distance_estimate=distance,
                        direction=direction,
                        confidence=det["confidence"],
                        growth_rate=growth,
                        bbox_area_ratio=area_ratio,
                    )
                )

    # ------------------------------------------------------------------
    # Scene summary
    # ------------------------------------------------------------------
    async def _emit_scene_summary(
        self, detections: list[dict], timestamp: float
    ) -> None:
        """Emit a periodic scene summary event."""
        counts: dict[str, int] = defaultdict(int)
        for det in detections:
            counts[det["class"]] += 1

        parts = [f"{v} {k}{'s' if v > 1 else ''}" for k, v in counts.items()]
        summary = (
            "Scene: " + ", ".join(parts) if parts else "No objects detected"
        )

        if self._events:
            self._events.send(
                SceneSummaryEvent(
                    summary=summary,
                    object_counts=dict(counts),
                    timestamp_unix=timestamp,
                )
            )

    # ------------------------------------------------------------------
    # Spatial estimation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_direction(center_x: float, frame_width: int) -> str:
        """Estimate object direction from horizontal bbox centre."""
        ratio = center_x / frame_width if frame_width > 0 else 0.5
        if ratio < 0.33:
            return "left"
        elif ratio > 0.66:
            return "right"
        return "centre"

    @staticmethod
    def _estimate_distance(area_ratio: float) -> str:
        """Estimate distance from bbox area ratio."""
        if area_ratio > DISTANCE_THRESHOLDS["near"]:
            return "near"
        elif area_ratio > DISTANCE_THRESHOLDS["medium"]:
            return "medium"
        return "far"

    # ------------------------------------------------------------------
    # Detection log (for spatial memory integration)
    # ------------------------------------------------------------------
    def _log_detections(self, detections: list[dict], timestamp: float):
        """Store detections locally and sync to spatial memory."""
        enriched = []
        for d in detections:
            direction = self._estimate_direction(d["center_x"], 1)
            distance = self._estimate_distance(d["area_ratio"])
            enriched.append({
                "class": d["class"],
                "confidence": d["confidence"],
                "direction": direction,
                "distance": distance,
                "frame_number": self._frame_count,
            })

        entry = {
            "frame": self._frame_count,
            "timestamp": timestamp,
            "objects": enriched,
        }
        self._detection_log.append(entry)
        if len(self._detection_log) > 500:
            self._detection_log = self._detection_log[-250:]

        # Day 4: Async log to spatial memory (non-blocking)
        if self._spatial_memory and enriched:
            asyncio.create_task(self._sync_to_memory(enriched))

        # Day 4: Process through navigation engine for smart announcements
        if self._navigation_engine and enriched:
            announcements = self._navigation_engine.process_detections(enriched)
            if announcements:
                for ann in announcements:
                    logger.info(
                        "NAV: [P%d] %s",
                        ann["priority"],
                        ann["text"],
                    )

    async def _sync_to_memory(self, detections: list[dict]) -> None:
        """Sync detections to spatial memory (async, fire-and-forget)."""
        try:
            logged = await self._spatial_memory.log_detection_batch(detections)
            if logged > 0:
                logger.debug(
                    "Synced %d detections to spatial memory", logged
                )
        except Exception as e:
            logger.debug("Spatial memory sync error: %s", e)

    @property
    def detection_history(self) -> list[dict]:
        """Recent detection history for spatial-memory queries."""
        return self._detection_log[-100:]

    # ------------------------------------------------------------------
    # Publishing & lifecycle
    # ------------------------------------------------------------------
    def publish_video_track(self) -> aiortc.VideoStreamTrack:
        """Publish the annotated video track with detection overlays."""
        return self._video_track

    async def stop_processing(self) -> None:
        if self._forwarder:
            await self._forwarder.remove_frame_handler(self._process_frame)
            self._forwarder = None

    async def close(self) -> None:
        await self.stop_processing()
        self._video_track.stop()
        self._bbox_tracker.clear()
        logger.info(
            "GuideLensProcessor closed (total frames: %d)", self._frame_count
        )
