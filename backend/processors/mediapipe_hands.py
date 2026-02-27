"""
MediaPipe Hand Landmarker — 21-keypoint hand tracking for SignBridge
=====================================================================
Provides fine-grained finger-level hand landmark detection using Google
MediaPipe's HandLandmarker Tasks API (v0.10.32+). This runs alongside
the existing YOLO Pose pipeline to augment the 17 COCO body keypoints
with 21 hand keypoints per hand, enabling finger-spelling and fine
gesture recognition.

Architecture:
  Video Frame → MediaPipe HandLandmarker → 21 keypoints × 2 hands
                     ↓
              Hand Skeleton Overlay (fingertips highlighted)
                     ↓
              FingerState analysis → per-finger extended/curled
                     ↓
              Basic ASL letter classification (A-Z static subset)

Integration:
  Called from SignBridgeProcessor._process_frame() when available.
  Results are merged into the gesture buffer for richer classification.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("signbridge.mediapipe_hands")

# ---------------------------------------------------------------------------
# Lazy MediaPipe import (module may not be installed)
# ---------------------------------------------------------------------------
_mp = None

try:
    import mediapipe as mp
    _mp = mp
    logger.info("MediaPipe %s loaded successfully", mp.__version__)
except ImportError:
    logger.warning("mediapipe not installed — hand landmark detection unavailable")


# ---------------------------------------------------------------------------
# 21 MediaPipe Hand Landmark Names
# ---------------------------------------------------------------------------
HAND_LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

# Fingertip indices (for highlighting)
FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips

# Finger PIP (proximal) indices — used for extended/curled detection
FINGER_PIP_INDICES = [2, 6, 10, 14, 18]  # thumb MCP, index PIP, middle PIP, ring PIP, pinky PIP

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# Hand skeleton connections for custom drawing (pairs of landmark indices)
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class HandResult:
    """Result from MediaPipe hand landmark detection for one hand."""
    handedness: str  # "Left" or "Right"
    confidence: float
    landmarks: list  # 21 (x, y, z) normalized landmarks
    pixel_landmarks: list  # 21 (x, y) pixel coordinates
    finger_states: dict = field(default_factory=dict)  # finger_name → "extended" | "curled"
    asl_letter: str = ""


@dataclass
class HandsDetectionResult:
    """Combined result from MediaPipe for all hands in frame."""
    hands: list  # List[HandResult]
    num_hands: int = 0
    annotated_image: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# MediaPipe Hand Landmarker (Tasks API)
# ---------------------------------------------------------------------------
class MediaPipeHandLandmarker:
    """
    Wraps Google MediaPipe HandLandmarker Tasks API for 21-keypoint hand
    landmark detection with finger state analysis and basic ASL letter
    recognition.
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: Optional[str] = None,
    ):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._model_path = model_path
        self._detector = None
        self._available = False
        self._total_detections = 0
        self._init()

    def _init(self):
        """Initialize MediaPipe HandLandmarker Tasks API."""
        if _mp is None:
            logger.warning("MediaPipe not available — hand landmarker disabled")
            return

        # Resolve model path
        model_path = self._model_path
        if model_path is None:
            # Look for model file relative to this module's directory, then backend root
            candidates = [
                os.path.join(os.path.dirname(__file__), "..", "hand_landmarker.task"),
                os.path.join(os.path.dirname(__file__), "hand_landmarker.task"),
                "hand_landmarker.task",
            ]
            for candidate in candidates:
                if os.path.isfile(candidate):
                    model_path = candidate
                    break

        if not model_path or not os.path.isfile(model_path):
            logger.warning(
                "hand_landmarker.task model file not found — hand detection disabled. "
                "Download from https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            )
            return

        try:
            base_options = _mp.tasks.BaseOptions(model_asset_path=model_path)
            options = _mp.tasks.vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=_mp.tasks.vision.RunningMode.IMAGE,
                num_hands=self.max_num_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._detector = _mp.tasks.vision.HandLandmarker.create_from_options(options)
            self._available = True
            logger.info(
                "MediaPipe HandLandmarker initialized (max_hands=%d, det_conf=%.2f, model=%s)",
                self.max_num_hands,
                self.min_detection_confidence,
                model_path,
            )
        except Exception as e:
            logger.error("Failed to initialize MediaPipe HandLandmarker: %s", e)

    @property
    def available(self) -> bool:
        return self._available

    def detect(self, image: np.ndarray, draw: bool = True) -> HandsDetectionResult:
        """
        Run hand landmark detection on an RGB image.

        Args:
            image: RGB numpy array (H, W, 3)
            draw: Whether to draw hand skeleton overlay on the image

        Returns:
            HandsDetectionResult with per-hand landmarks and finger states
        """
        if not self._available or self._detector is None:
            return HandsDetectionResult(hands=[], num_hands=0)

        h, w, _ = image.shape

        # Convert to MediaPipe Image
        mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=image)
        results = self._detector.detect(mp_image)

        hands: list[HandResult] = []
        annotated = image.copy() if draw else None

        if results.hand_landmarks and results.handedness:
            for hand_landmarks, handedness_list in zip(
                results.hand_landmarks,
                results.handedness,
            ):
                # Extract handedness (Tasks API returns list of Category)
                label = handedness_list[0].category_name  # "Left" or "Right"
                conf = handedness_list[0].score

                # Convert normalized landmarks to pixel coordinates
                normalized = []
                pixel_coords = []
                for lm in hand_landmarks:
                    normalized.append((lm.x, lm.y, lm.z))
                    pixel_coords.append((int(lm.x * w), int(lm.y * h)))

                # Analyze finger states
                finger_states = self._analyze_fingers(pixel_coords, label)

                # Classify ASL letter (static hand shapes only)
                asl_letter = self._classify_asl_static(finger_states, pixel_coords)

                hand = HandResult(
                    handedness=label,
                    confidence=conf,
                    landmarks=normalized,
                    pixel_landmarks=pixel_coords,
                    finger_states=finger_states,
                    asl_letter=asl_letter,
                )
                hands.append(hand)
                self._total_detections += 1

                # Draw overlay (custom drawing since Tasks API doesn't use solutions drawing)
                if draw and annotated is not None:
                    self._draw_hand(annotated, pixel_coords, finger_states, label, asl_letter)

        return HandsDetectionResult(
            hands=hands,
            num_hands=len(hands),
            annotated_image=annotated,
        )

    def _draw_hand(
        self,
        image: np.ndarray,
        pixel_coords: list,
        finger_states: dict,
        label: str,
        asl_letter: str,
    ):
        """Draw hand skeleton, fingertip highlights, and labels on the image."""
        # Draw connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(pixel_coords) and end_idx < len(pixel_coords):
                cv2.line(
                    image,
                    pixel_coords[start_idx],
                    pixel_coords[end_idx],
                    (200, 200, 200),  # light gray
                    2,
                )

        # Draw all landmarks as small circles
        for idx, (px, py) in enumerate(pixel_coords):
            cv2.circle(image, (px, py), 3, (255, 255, 255), -1)

        # Highlight fingertips with color based on state
        for i, tip_idx in enumerate(FINGERTIP_INDICES):
            if tip_idx < len(pixel_coords):
                px, py = pixel_coords[tip_idx]
                finger_name = FINGER_NAMES[i]
                state = finger_states.get(finger_name, "unknown")
                color = (0, 255, 0) if state == "extended" else (0, 0, 255)
                cv2.circle(image, (px, py), 6, color, -1)

        # Show handedness + ASL letter label
        wrist_px, wrist_py = pixel_coords[0]
        label_text = label
        if asl_letter:
            label_text += f" [{asl_letter}]"
        cv2.putText(
            image,
            label_text,
            (wrist_px - 20, wrist_py - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

    def _analyze_fingers(
        self, pixel_coords: list, handedness: str
    ) -> dict[str, str]:
        """
        Determine whether each finger is extended or curled.

        Uses the simple heuristic: fingertip is above (lower y) its PIP joint
        → extended, otherwise → curled. Thumb uses x-axis comparison based
        on handedness.
        """
        states: dict[str, str] = {}

        if len(pixel_coords) < 21:
            return {name: "unknown" for name in FINGER_NAMES}

        # Thumb: compare tip x relative to IP joint based on handedness
        thumb_tip = pixel_coords[4]
        thumb_ip = pixel_coords[3]
        if handedness == "Right":
            states["thumb"] = "extended" if thumb_tip[0] < thumb_ip[0] else "curled"
        else:
            states["thumb"] = "extended" if thumb_tip[0] > thumb_ip[0] else "curled"

        # Index, middle, ring, pinky: tip y < PIP y means extended
        for finger_name, tip_idx, pip_idx in [
            ("index", 8, 6),
            ("middle", 12, 10),
            ("ring", 16, 14),
            ("pinky", 20, 18),
        ]:
            states[finger_name] = (
                "extended"
                if pixel_coords[tip_idx][1] < pixel_coords[pip_idx][1]
                else "curled"
            )

        return states

    def _classify_asl_static(
        self, finger_states: dict, pixel_coords: list
    ) -> str:
        """
        Classify basic static ASL finger-spelling letters.
        Only covers letters that are single static hand poses
        (not motion-based like J, Z).
        """
        if not finger_states or len(pixel_coords) < 21:
            return ""

        t = finger_states.get("thumb", "")
        i = finger_states.get("index", "")
        m = finger_states.get("middle", "")
        r = finger_states.get("ring", "")
        p = finger_states.get("pinky", "")

        extended = [t, i, m, r, p]
        num_extended = extended.count("extended")

        # A: fist with thumb alongside (all curled, thumb extended/alongside)
        if i == m == r == p == "curled" and t == "extended":
            return "A"

        # B: all fingers extended, thumb curled across palm
        if i == m == r == p == "extended" and t == "curled":
            return "B"

        # D: index extended, others curled, thumb touches middle finger
        if i == "extended" and m == r == p == "curled" and t == "curled":
            return "D"

        # I: pinky extended, rest curled
        if p == "extended" and i == m == r == "curled":
            return "I"

        # L: index + thumb extended (L shape)
        if i == "extended" and t == "extended" and m == r == p == "curled":
            return "L"

        # V/Peace: index + middle extended
        if i == m == "extended" and r == p == "curled":
            return "V"

        # W: index + middle + ring extended
        if i == m == r == "extended" and p == "curled":
            return "W"

        # Y: thumb + pinky extended (hang loose)
        if t == "extended" and p == "extended" and i == m == r == "curled":
            return "Y"

        # 5/Open hand: all extended
        if num_extended == 5:
            return "5"

        # Fist: all curled
        if num_extended == 0:
            return "S"

        return ""

    def get_telemetry(self) -> dict:
        """Return telemetry data for this detector."""
        return {
            "detector": "mediapipe_hands",
            "available": self._available,
            "max_hands": self.max_num_hands,
            "total_detections": self._total_detections,
        }

    def close(self):
        """Release MediaPipe resources."""
        if self._detector:
            self._detector.close()
            self._detector = None
            self._available = False
            logger.info("MediaPipe HandLandmarker closed")
