"""
SignBridge Processor — YOLO Pose + MediaPipe Hands for Sign Language
=====================================================================
Day 2+5: Full implementation with YOLOv11 Pose + Google MediaPipe Hand
Landmarks + skeletal extraction + gesture buffering + HuggingFace NLP.

Architecture:
  Video Frame → YOLO Pose → 17 Body Keypoints → Gesture Buffer → Event Emission
       ↓
  MediaPipe Hands → 21 Hand Keypoints × 2 → Finger State → ASL Letter
       ↓
  Combined Skeleton + Hand Overlay → Published Video Track

Features:
  - Extracts 17 COCO skeletal keypoints per person (YOLO Pose)
  - Extracts 21 hand landmarks per hand with finger state analysis (MediaPipe)
  - Draws upper-body focused skeleton overlay with wrist highlights
  - Highlights individual fingertips (green=extended, red=curled)
  - Basic ASL finger-spelling recognition (A, B, D, I, L, V, W, Y, S, 5)
  - `GestureBuffer` with temporal motion analysis (30-frame window)
  - Rule-based gesture classifier (WAVE, RAISE-HAND, POINT, ACTIVE-SIGN)
  - `GlossTranslator` with optional HuggingFace Inference API
  - Emits structured events with hand landmark + finger state data
"""

import asyncio
import logging
import os
import time
from collections import deque
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

from processors.mediapipe_hands import MediaPipeHandLandmarker

logger = logging.getLogger("signbridge.processor")


# ---------------------------------------------------------------------------
# COCO 17 Keypoint Topology
# ---------------------------------------------------------------------------
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Upper-body skeleton connections (most relevant for sign language)
UPPER_BODY_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),    # Head
    (5, 6),                              # Shoulders
    (5, 7), (7, 9),                      # Left arm
    (6, 8), (8, 10),                     # Right arm
    (5, 11), (6, 12),                    # Torso top
]

# Full skeleton (for reference — we draw upper body only for sign language)
FULL_SKELETON_CONNECTIONS = UPPER_BODY_CONNECTIONS + [
    (11, 12),                            # Hips
    (11, 13), (13, 15),                  # Left leg
    (12, 14), (14, 16),                  # Right leg
]

# Key indices for sign language analysis
HAND_KEYPOINTS = [9, 10]           # left_wrist, right_wrist
ARM_KEYPOINTS = [5, 6, 7, 8, 9, 10]  # shoulders, elbows, wrists
HEAD_KEYPOINTS = [0, 1, 2, 3, 4]     # nose, eyes, ears


# ---------------------------------------------------------------------------
# Custom Events
# ---------------------------------------------------------------------------
@dataclass
class SignDetectedEvent(BaseEvent):
    """Emitted when skeletal keypoints are detected in a frame."""
    type: str = field(default="signbridge.sign_detected", init=False)
    keypoints: list = field(default_factory=list)
    num_persons: int = 0
    confidence: float = 0.0
    frame_number: int = 0
    timestamp_unix: float = 0.0
    # Day 5: MediaPipe hand landmarks
    num_hands: int = 0
    finger_states: list = field(default_factory=list)  # per-hand finger state dicts
    asl_letters: list = field(default_factory=list)  # per-hand detected ASL letters


@dataclass
class GestureBufferEvent(BaseEvent):
    """Emitted when enough frames are buffered for gesture analysis."""
    type: str = field(default="signbridge.gesture_buffer", init=False)
    gesture_sequence: list = field(default_factory=list)
    buffer_length: int = 0
    raw_gloss: str = ""


@dataclass
class SignTranslationEvent(BaseEvent):
    """Emitted when raw sign gloss is translated to fluent text."""
    type: str = field(default="signbridge.sign_translation", init=False)
    raw_gloss: str = ""
    translated_text: str = ""
    language: str = "en"


# ---------------------------------------------------------------------------
# Gesture Buffer — temporal sequence analysis
# ---------------------------------------------------------------------------
class GestureBuffer:
    """
    Buffers keypoint sequences for temporal gesture recognition.
    Analyzes wrist trajectories to detect significant signing motion
    and classify basic gesture patterns.
    """

    def __init__(self, max_frames: int = 30, min_frames_for_gesture: int = 10):
        self.max_frames = max_frames
        self.min_frames = min_frames_for_gesture
        self._buffer: deque = deque(maxlen=max_frames)
        self._last_emit_time = 0.0
        self._cooldown = 2.0  # seconds between gesture emissions

    def add_frame(self, keypoints: list, timestamp: float) -> Optional[list]:
        """
        Add keypoints for one frame.
        Returns the gesture sequence if sufficient motion is detected and
        the cooldown period has elapsed.
        """
        self._buffer.append({
            "keypoints": keypoints,
            "timestamp": timestamp,
        })

        if (
            len(self._buffer) >= self.min_frames
            and timestamp - self._last_emit_time > self._cooldown
            and self._has_significant_motion()
        ):
            self._last_emit_time = timestamp
            return list(self._buffer)
        return None

    def _has_significant_motion(self) -> bool:
        """Detect significant upper-body motion across the buffer window."""
        if len(self._buffer) < 2:
            return False

        first = self._buffer[0]["keypoints"]
        last = self._buffer[-1]["keypoints"]

        if not first or not last:
            return False

        try:
            first_kps = first[0] if first else []
            last_kps = last[0] if last else []

            if len(first_kps) < 11 or len(last_kps) < 11:
                return False

            # Measure total wrist displacement
            total_displacement = 0.0
            for idx in HAND_KEYPOINTS:
                if idx < len(first_kps) and idx < len(last_kps):
                    fx, fy = first_kps[idx][:2]
                    lx, ly = last_kps[idx][:2]
                    total_displacement += ((lx - fx) ** 2 + (ly - fy) ** 2) ** 0.5

            return total_displacement > 50  # pixel threshold
        except (IndexError, TypeError):
            return False

    def clear(self):
        self._buffer.clear()

    @property
    def length(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# HuggingFace Gloss Translator (optional)
# ---------------------------------------------------------------------------
class GlossTranslator:
    """
    Translates raw sign-language glosses to fluent English.

    If HF_TOKEN is set, uses the HuggingFace Inference API.
    Otherwise, falls back to a simple rule-based mapping.
    """

    def __init__(self):
        self._client = None
        self._hf_available = False
        self._init_hf()

    def _init_hf(self):
        """Try to initialise HuggingFace Inference client."""
        hf_token = os.getenv("HF_TOKEN", "")
        if hf_token and not hf_token.startswith("your_"):
            try:
                from huggingface_hub import InferenceClient
                self._client = InferenceClient(token=hf_token)
                self._hf_available = True
                logger.info("HuggingFace NLP translator initialised")
            except Exception as e:
                logger.warning("HuggingFace client init failed: %s — using rule-based fallback", e)
        else:
            logger.info("HF_TOKEN not set — using rule-based gloss translation")

    # Simple gloss → English mappings for common gestures
    _GLOSS_MAP = {
        "WAVE": "Hello! (waving gesture detected)",
        "RAISE-HAND": "Raising hand — attention or question gesture",
        "POINT-RIGHT": "Pointing to the right",
        "POINT-LEFT": "Pointing to the left",
        "ACTIVE-SIGN": "Active signing detected — multiple hand movements",
        "GESTURE": "Gesture detected",
        "BOTH-HANDS-UP": "Both hands raised",
        "CROSS-ARMS": "Arms crossed",
    }

    async def translate(self, raw_gloss: str) -> str:
        """Translate gloss to fluent English."""
        if not raw_gloss:
            return ""

        # Try HuggingFace first
        if self._hf_available and self._client:
            try:
                prompt = (
                    "You are a sign language interpreter. Convert the following "
                    "sign-language gloss notation into fluent, natural English. "
                    "Keep your response to one sentence.\n\n"
                    f"Gloss: {raw_gloss}\n"
                    "English:"
                )
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self._client.text_generation(
                        prompt,
                        model="meta-llama/Meta-Llama-3-8B-Instruct",
                        max_new_tokens=60,
                    ),
                )
                translated = result.strip()
                if translated:
                    return translated
            except Exception as e:
                logger.warning("HuggingFace translation failed: %s — falling back", e)

        # Rule-based fallback
        return self._GLOSS_MAP.get(raw_gloss, f"Sign gesture: {raw_gloss}")


# ---------------------------------------------------------------------------
# SignBridge Processor
# ---------------------------------------------------------------------------
class SignBridgeProcessor(VideoProcessorPublisher):
    """
    Full YOLO Pose processor for sign language detection.

    Receives video frames, runs YOLOv11 Pose estimation, extracts skeletal
    keypoints, buffers gesture sequences, and publishes annotated frames
    with skeleton overlay emphasising upper-body and wrist movements.
    """

    name = "signbridge_pose"

    def __init__(
        self,
        fps: int = 10,
        conf_threshold: float = 0.5,
        model_path: str = "yolo11n-pose.pt",
        device: str = "cpu",
        gesture_buffer_frames: int = 30,
        enable_mediapipe_hands: bool = True,
    ):
        self.fps = fps
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        self.device = device

        self._forwarder: Optional[VideoForwarder] = None
        self._video_track = QueuedVideoTrack()
        self._frame_count = 0
        self._model = None
        self._gesture_buffer = GestureBuffer(
            max_frames=gesture_buffer_frames,
            min_frames_for_gesture=10,
        )
        self._translator = GlossTranslator()
        self._agent = None
        self._events = None
        self._processing = False  # inference mutex

        # Day 5: MediaPipe Hand Landmarker (21 keypoints per hand)
        self._hand_landmarker: Optional[MediaPipeHandLandmarker] = None
        if enable_mediapipe_hands:
            try:
                self._hand_landmarker = MediaPipeHandLandmarker(
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                if self._hand_landmarker.available:
                    logger.info("MediaPipe Hand Landmarker enabled for SignBridge")
                else:
                    logger.warning("MediaPipe not available — running YOLO Pose only")
                    self._hand_landmarker = None
            except Exception as e:
                logger.warning("MediaPipe init failed: %s — YOLO Pose only", e)
                self._hand_landmarker = None

        # Day 5: Telemetry metrics
        self._total_gestures_detected = 0
        self._total_persons_detected = 0
        self._total_hands_detected = 0
        self._total_asl_letters_detected = 0
        self._total_inference_time = 0.0
        self._inference_count = 0
        self._start_time = time.time()

        # Load YOLO model
        self._load_model()

    def _load_model(self):
        """Load the YOLO pose estimation model."""
        try:
            from ultralytics import YOLO

            logger.info(
                "Loading YOLO pose model: %s (device: %s)",
                self.model_path,
                self.device,
            )
            self._model = YOLO(self.model_path)
            logger.info("YOLO pose model loaded successfully")
        except Exception as e:
            logger.error("Failed to load YOLO model: %s", e)
            logger.warning(
                "SignBridge will run in pass-through mode (no pose detection)"
            )

    # ------------------------------------------------------------------
    # Agent lifecycle
    # ------------------------------------------------------------------
    def attach_agent(self, agent):
        """Register custom events with the agent's event system."""
        self._agent = agent
        self._events = agent.events
        self._events.register(SignDetectedEvent)
        self._events.register(GestureBufferEvent)
        self._events.register(SignTranslationEvent)
        logger.info("SignBridgeProcessor attached to agent")

    def get_telemetry(self) -> dict:
        """Return real-time telemetry metrics for the /telemetry endpoint."""
        avg_inference = (
            (self._total_inference_time / self._inference_count * 1000)
            if self._inference_count > 0 else 0.0
        )
        telemetry = {
            "processor": "signbridge_pose",
            "model": self.model_path,
            "device": self.device,
            "target_fps": self.fps,
            "frames_processed": self._frame_count,
            "total_gestures_detected": self._total_gestures_detected,
            "total_persons_detected": self._total_persons_detected,
            "total_hands_detected": self._total_hands_detected,
            "total_asl_letters_detected": self._total_asl_letters_detected,
            "avg_inference_ms": round(avg_inference, 1),
            "inference_count": self._inference_count,
            "gesture_buffer_size": self._gesture_buffer.length,
            "mediapipe_enabled": self._hand_landmarker is not None,
            "uptime_seconds": round(time.time() - self._start_time, 1),
        }
        if self._hand_landmarker:
            telemetry["mediapipe"] = self._hand_landmarker.get_telemetry()
        return telemetry

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
            name="signbridge_pose",
        )
        logger.info(
            "SignBridgeProcessor started (fps=%d, conf=%.2f, model=%s)",
            self.fps,
            self.conf_threshold,
            self.model_path,
        )

    # ------------------------------------------------------------------
    # Frame processing pipeline
    # ------------------------------------------------------------------
    async def _process_frame(self, frame: av.VideoFrame) -> None:
        """Process a single video frame with YOLO pose estimation."""
        # Skip if a previous frame is still being processed
        if self._processing:
            await self._video_track.add_frame(frame)
            return

        self._processing = True
        try:
            self._frame_count += 1
            timestamp = time.time()

            img = frame.to_ndarray(format="rgb24")

            if self._model is not None:
                t_inference_start = time.time()
                annotated_img, all_keypoints = await self._run_pose_detection(img)
                t_inference_end = time.time()
                self._total_inference_time += (t_inference_end - t_inference_start)
                self._inference_count += 1

                # Day 5: Run MediaPipe hand detection on the same frame
                hand_data = None
                if self._hand_landmarker and self._hand_landmarker.available:
                    hand_data = self._hand_landmarker.detect(annotated_img, draw=True)
                    if hand_data.annotated_image is not None:
                        annotated_img = hand_data.annotated_image
                    self._total_hands_detected += hand_data.num_hands
                    # Count ASL letters detected
                    for hand in hand_data.hands:
                        if hand.asl_letter:
                            self._total_asl_letters_detected += 1

                # Emit detection event
                if all_keypoints and self._events:
                    self._total_persons_detected += len(all_keypoints)
                    event = SignDetectedEvent(
                        keypoints=all_keypoints,
                        num_persons=len(all_keypoints),
                        confidence=self._avg_confidence(all_keypoints),
                        frame_number=self._frame_count,
                        timestamp_unix=timestamp,
                    )
                    # Enrich with hand landmark data
                    if hand_data and hand_data.num_hands > 0:
                        event.num_hands = hand_data.num_hands
                        event.finger_states = [h.finger_states for h in hand_data.hands]
                        event.asl_letters = [h.asl_letter for h in hand_data.hands if h.asl_letter]
                    self._events.send(event)

                # Gesture buffering + classification
                gesture_seq = self._gesture_buffer.add_frame(
                    all_keypoints, timestamp
                )
                if gesture_seq and self._events:
                    raw_gloss = self._classify_gesture(gesture_seq)

                    # Day 5: Enhance gloss with ASL letter from MediaPipe
                    if hand_data and hand_data.num_hands > 0:
                        asl_letters = [h.asl_letter for h in hand_data.hands if h.asl_letter]
                        if asl_letters and not raw_gloss:
                            raw_gloss = f"FINGERSPELL-{'-'.join(asl_letters)}"
                        elif asl_letters:
                            raw_gloss = f"{raw_gloss} [{','.join(asl_letters)}]"

                    self._total_gestures_detected += 1
                    self._events.send(
                        GestureBufferEvent(
                            gesture_sequence=[],  # omit heavy data
                            buffer_length=len(gesture_seq),
                            raw_gloss=raw_gloss,
                        )
                    )

                    if raw_gloss:
                        translated = await self._translator.translate(raw_gloss)
                        if translated:
                            self._events.send(
                                SignTranslationEvent(
                                    raw_gloss=raw_gloss,
                                    translated_text=translated,
                                    language="en",
                                )
                            )

                new_frame = av.VideoFrame.from_ndarray(annotated_img, format="rgb24")
            else:
                new_frame = frame

            await self._video_track.add_frame(new_frame)

            if self._frame_count % 100 == 0:
                logger.info(
                    "SignBridge: %d frames processed (gesture buffer: %d)",
                    self._frame_count,
                    self._gesture_buffer.length,
                )
        finally:
            self._processing = False

    async def _run_pose_detection(
        self, img: np.ndarray
    ) -> tuple[np.ndarray, list]:
        """
        Run YOLO pose detection in a thread-pool executor.
        Returns (annotated_image, list_of_person_keypoints).
        Each person's keypoints is a list of (x, y) tuples for the 17 COCO points.
        """
        loop = asyncio.get_event_loop()

        def _detect():
            results = self._model(
                img,
                conf=self.conf_threshold,
                verbose=False,
                device=self.device,
            )

            all_keypoints: list[list[tuple[float, float]]] = []
            annotated = img.copy()

            for result in results:
                if result.keypoints is None or result.keypoints.xy is None:
                    continue

                for person_kps in result.keypoints.xy:
                    kps_np = person_kps.cpu().numpy()
                    person_keypoints = [
                        (float(kp[0]), float(kp[1])) for kp in kps_np
                    ]
                    all_keypoints.append(person_keypoints)

                annotated = self._draw_skeleton(annotated, all_keypoints)

            return annotated, all_keypoints

        return await loop.run_in_executor(None, _detect)

    # ------------------------------------------------------------------
    # Skeleton drawing
    # ------------------------------------------------------------------
    def _draw_skeleton(
        self, img: np.ndarray, all_keypoints: list
    ) -> np.ndarray:
        """Draw skeleton overlay on the image — upper body focus."""
        palette = [
            (102, 99, 255),   # purple
            (255, 178, 102),  # orange
            (102, 255, 178),  # green
            (255, 102, 178),  # pink
        ]
        wrist_color = (0, 255, 255)  # yellow

        for pidx, keypoints in enumerate(all_keypoints):
            color = palette[pidx % len(palette)]

            # Draw connections
            for si, ei in UPPER_BODY_CONNECTIONS:
                if si < len(keypoints) and ei < len(keypoints):
                    x1, y1 = int(keypoints[si][0]), int(keypoints[si][1])
                    x2, y2 = int(keypoints[ei][0]), int(keypoints[ei][1])
                    if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                        cv2.line(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Draw keypoints
            for kp_idx, (x, y) in enumerate(keypoints):
                ix, iy = int(x), int(y)
                if ix <= 0 or iy <= 0:
                    continue
                if kp_idx in HAND_KEYPOINTS:
                    cv2.circle(img, (ix, iy), 8, wrist_color, -1, cv2.LINE_AA)
                    cv2.circle(img, (ix, iy), 10, wrist_color, 2, cv2.LINE_AA)
                elif kp_idx in ARM_KEYPOINTS:
                    cv2.circle(img, (ix, iy), 5, color, -1, cv2.LINE_AA)
                else:
                    cv2.circle(img, (ix, iy), 3, color, -1, cv2.LINE_AA)

        # HUD overlay
        cv2.putText(
            img,
            f"SignBridge | Persons: {len(all_keypoints)} | Frame: {self._frame_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        return img

    # ------------------------------------------------------------------
    # Gesture classification (rule-based — future: trained model)
    # ------------------------------------------------------------------
    def _classify_gesture(self, gesture_sequence: list) -> str:
        """
        Classify a buffered gesture sequence into a raw sign gloss.
        Uses wrist trajectory heuristics. In production, replace with
        a proper gesture-recognition model.
        """
        if not gesture_sequence or len(gesture_sequence) < 5:
            return ""

        try:
            first_kps = (gesture_sequence[0]["keypoints"] or [[]])[0]
            last_kps = (gesture_sequence[-1]["keypoints"] or [[]])[0]
            if len(first_kps) < 11 or len(last_kps) < 11:
                return ""

            lw_dx = last_kps[9][0] - first_kps[9][0]
            lw_dy = last_kps[9][1] - first_kps[9][1]
            rw_dx = last_kps[10][0] - first_kps[10][0]
            rw_dy = last_kps[10][1] - first_kps[10][1]

            # Both hands raised
            if lw_dy < -80 and rw_dy < -80:
                return "BOTH-HANDS-UP"

            # Horizontal wave
            if (abs(lw_dx) > 80 or abs(rw_dx) > 80) and (
                abs(lw_dy) < 40 and abs(rw_dy) < 40
            ):
                return "WAVE"

            # Single hand raise
            if lw_dy < -60 or rw_dy < -60:
                return "RAISE-HAND"

            # Pointing
            if rw_dx > 60 and abs(rw_dy) < 30:
                return "POINT-RIGHT"
            if lw_dx < -60 and abs(lw_dy) < 30:
                return "POINT-LEFT"

            # Active two-hand signing
            lm = (lw_dx**2 + lw_dy**2) ** 0.5
            rm = (rw_dx**2 + rw_dy**2) ** 0.5
            if lm > 40 and rm > 40:
                return "ACTIVE-SIGN"
            if lm > 40 or rm > 40:
                return "GESTURE"

            return ""
        except (IndexError, TypeError, ValueError):
            return ""

    @staticmethod
    def _avg_confidence(all_keypoints: list) -> float:
        """Average confidence (YOLO already filtered by conf_threshold)."""
        return 1.0 if all_keypoints else 0.0

    # ------------------------------------------------------------------
    # Publishing & lifecycle
    # ------------------------------------------------------------------
    def publish_video_track(self) -> aiortc.VideoStreamTrack:
        """Publish the annotated video track with skeleton overlay."""
        return self._video_track

    async def stop_processing(self) -> None:
        if self._forwarder:
            await self._forwarder.remove_frame_handler(self._process_frame)
            self._forwarder = None

    async def close(self) -> None:
        await self.stop_processing()
        self._video_track.stop()
        self._gesture_buffer.clear()
        if self._hand_landmarker:
            self._hand_landmarker.close()
        logger.info(
            "SignBridgeProcessor closed (total frames: %d, hands: %d, asl_letters: %d)",
            self._frame_count,
            self._total_hands_detected,
            self._total_asl_letters_detected,
        )
