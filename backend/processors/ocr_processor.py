"""
OCR / VLM Processor — Text Recognition & Dense Scene Descriptions
==================================================================
Day 3: Provides on-demand OCR and scene captioning via the multi-provider
fallback chain (Gemini → Grok → Azure → NVIDIA → HuggingFace).

This processor runs alongside GuideLens or independently. It does NOT run
OCR on every frame (too expensive). Instead it:

  1. Captures recent frames from the shared VideoForwarder.
  2. Exposes methods that MCP tools call on-demand when the user asks
     "What does that sign say?" or "Describe the scene in detail".
  3. Runs a lightweight periodic scan (every N seconds) looking for
     text-heavy regions and caching results for the frontend overlay.

The processor stores an internal frame buffer and OCR result cache so
that the frontend can poll `/ocr-results` for overlay data.
"""

import asyncio
import io
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import av
import aiortc
import cv2
import numpy as np

from vision_agents.core.events import BaseEvent
from vision_agents.core.processors import VideoProcessor
from vision_agents.core.utils.video_forwarder import VideoForwarder

logger = logging.getLogger("worldlens.ocr")


# ---------------------------------------------------------------------------
# Custom Events
# ---------------------------------------------------------------------------
@dataclass
class OCRResultEvent(BaseEvent):
    """Emitted when OCR text is extracted from a frame."""
    type: str = field(default="ocr.text_detected", init=False)
    text: str = ""
    confidence: float = 0.0
    provider: str = ""
    frame_number: int = 0
    timestamp_unix: float = 0.0


@dataclass
class SceneDescriptionEvent(BaseEvent):
    """Emitted when a detailed scene description is generated."""
    type: str = field(default="ocr.scene_description", init=False)
    description: str = ""
    provider: str = ""
    timestamp_unix: float = 0.0


# ---------------------------------------------------------------------------
# OCR Processor
# ---------------------------------------------------------------------------
class OCRProcessor(VideoProcessor):
    """
    Captures video frames for on-demand OCR and captioning.

    This is a VideoProcessor (receive-only) — it does not publish a
    modified video track. Its job is to:
      1. Keep a rolling buffer of recent frames.
      2. Run periodic background OCR scans.
      3. Provide `read_text()` and `describe_scene()` methods for MCP tools.
    """

    name = "ocr_vlm"

    def __init__(
        self,
        scan_interval: float = 15.0,
        max_cached_results: int = 20,
        fps: int = 1,
    ):
        self.scan_interval = scan_interval
        self.max_cached_results = max_cached_results
        self.fps = fps

        self._forwarder: Optional[VideoForwarder] = None
        self._agent = None
        self._events = None
        self._frame_count = 0
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_time: float = 0.0
        self._frame_lock = asyncio.Lock()

        # OCR result cache — polled by frontend
        self._ocr_results: deque[dict] = deque(maxlen=max_cached_results)

        # Background scan state
        self._scan_task: Optional[asyncio.Task] = None
        self._running = False

        # Provider manager reference — set by main.py after creation
        self._provider_manager = None

    def set_provider_manager(self, pm):
        """Inject the ProviderManager for VLM calls."""
        self._provider_manager = pm

    # ------------------------------------------------------------------
    # Agent lifecycle
    # ------------------------------------------------------------------
    def attach_agent(self, agent):
        self._agent = agent
        self._events = agent.events
        self._events.register(OCRResultEvent)
        self._events.register(SceneDescriptionEvent)
        logger.info("OCRProcessor attached to agent")

    async def process_video(
        self,
        track: aiortc.VideoStreamTrack,
        participant_id: Optional[str],
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        """Start capturing frames (no output — read-only processor)."""
        if self._forwarder:
            await self._forwarder.remove_frame_handler(self._capture_frame)

        self._forwarder = shared_forwarder
        self._forwarder.add_frame_handler(
            self._capture_frame,
            fps=float(self.fps),
            name="ocr_frame_capture",
        )
        self._running = True

        # Start background OCR scan loop
        if self._scan_task is None or self._scan_task.done():
            self._scan_task = asyncio.create_task(self._background_scan_loop())

        logger.info(
            "OCRProcessor started (scan_interval=%.1fs, fps=%d)",
            self.scan_interval,
            self.fps,
        )

    async def _capture_frame(self, frame: av.VideoFrame) -> None:
        """Store the latest frame for on-demand processing."""
        self._frame_count += 1
        img = frame.to_ndarray(format="rgb24")
        async with self._frame_lock:
            self._latest_frame = img
            self._latest_frame_time = time.time()

    # ------------------------------------------------------------------
    # On-demand OCR (called by MCP tools)
    # ------------------------------------------------------------------
    async def read_text(self, prompt: str = "") -> dict:
        """
        Read text visible in the most recent frame.
        Returns a dict with 'text', 'provider', 'timestamp'.
        """
        img = await self._get_latest_frame()
        if img is None:
            return {
                "text": "",
                "error": "No video frame available",
                "provider": "none",
            }

        if not self._provider_manager:
            return {
                "text": "",
                "error": "Provider manager not configured",
                "provider": "none",
            }

        ocr_prompt = prompt or (
            "Read ALL text visible in this image. Include signs, labels, "
            "bus numbers, street names, notices, and any other text. "
            "Return ONLY the text you can read, formatted clearly. "
            "If no text is visible, say 'No text detected'."
        )

        try:
            jpeg_bytes = self._frame_to_jpeg(img)
            caption, provider = await self._provider_manager.caption(
                jpeg_bytes, ocr_prompt
            )

            result = {
                "text": caption.strip(),
                "provider": provider.value,
                "timestamp": time.time(),
                "frame_number": self._frame_count,
            }

            # Cache the result
            self._ocr_results.append(result)

            # Emit event
            if self._events:
                self._events.send(
                    OCRResultEvent(
                        text=caption.strip(),
                        provider=provider.value,
                        frame_number=self._frame_count,
                        timestamp_unix=time.time(),
                    )
                )

            logger.info(
                "OCR read_text via %s: %s",
                provider.value,
                caption[:80] + "..." if len(caption) > 80 else caption,
            )
            return result

        except Exception as e:
            logger.error("OCR read_text failed: %s", e)
            return {"text": "", "error": str(e), "provider": "none"}

    async def describe_scene(self, prompt: str = "") -> dict:
        """
        Generate a detailed scene description from the current frame.
        Uses NVIDIA/Gemini for dense visual reasoning.
        """
        img = await self._get_latest_frame()
        if img is None:
            return {
                "description": "",
                "error": "No video frame available",
                "provider": "none",
            }

        if not self._provider_manager:
            return {
                "description": "",
                "error": "Provider manager not configured",
                "provider": "none",
            }

        desc_prompt = prompt or (
            "Provide a detailed description of this scene for a visually "
            "impaired person. Include: spatial layout, objects and their "
            "positions, any text or signage, potential obstacles, lighting "
            "conditions, and any notable features. Be thorough but concise."
        )

        try:
            jpeg_bytes = self._frame_to_jpeg(img)
            description, provider = await self._provider_manager.caption(
                jpeg_bytes, desc_prompt
            )

            result = {
                "description": description.strip(),
                "provider": provider.value,
                "timestamp": time.time(),
            }

            # Emit event
            if self._events:
                self._events.send(
                    SceneDescriptionEvent(
                        description=description.strip(),
                        provider=provider.value,
                        timestamp_unix=time.time(),
                    )
                )

            logger.info(
                "Scene described via %s (%d chars)",
                provider.value,
                len(description),
            )
            return result

        except Exception as e:
            logger.error("describe_scene failed: %s", e)
            return {"description": "", "error": str(e), "provider": "none"}

    # ------------------------------------------------------------------
    # Background periodic OCR scan
    # ------------------------------------------------------------------
    async def _background_scan_loop(self):
        """Periodically scan frames for visible text."""
        logger.info(
            "Background OCR scan started (interval=%.1fs)", self.scan_interval
        )
        while self._running:
            await asyncio.sleep(self.scan_interval)
            if not self._running:
                break

            img = await self._get_latest_frame()
            if img is None or not self._provider_manager:
                continue

            try:
                jpeg_bytes = self._frame_to_jpeg(img)
                text, provider = await self._provider_manager.caption(
                    jpeg_bytes,
                    "List any text visible in this image: signs, labels, "
                    "numbers, or writing. Be brief. If none, say 'none'.",
                )
                text = text.strip().lower()
                if text and text != "none" and "no text" not in text:
                    result = {
                        "text": text,
                        "provider": provider.value,
                        "timestamp": time.time(),
                        "source": "background_scan",
                        "frame_number": self._frame_count,
                    }
                    self._ocr_results.append(result)

                    if self._events:
                        self._events.send(
                            OCRResultEvent(
                                text=text,
                                provider=provider.value,
                                frame_number=self._frame_count,
                                timestamp_unix=time.time(),
                            )
                        )
                    logger.info("Background OCR: '%s' via %s", text[:60], provider.value)
            except Exception as e:
                logger.debug("Background OCR scan skipped: %s", e)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    async def _get_latest_frame(self) -> Optional[np.ndarray]:
        """Get a copy of the most recent captured frame."""
        async with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    @staticmethod
    def _frame_to_jpeg(img: np.ndarray, quality: int = 85) -> bytes:
        """Convert an RGB numpy array to JPEG bytes."""
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()

    @property
    def cached_results(self) -> list[dict]:
        """Return cached OCR results for the frontend overlay."""
        return list(self._ocr_results)

    def get_recent_results(self, since: float = 0, limit: int = 10) -> list[dict]:
        """Get OCR results since a timestamp."""
        results = [r for r in self._ocr_results if r.get("timestamp", 0) > since]
        return results[-limit:]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def stop_processing(self) -> None:
        """Stop frame capture and background scan."""
        self._running = False
        if self._scan_task and not self._scan_task.done():
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
        if self._forwarder:
            await self._forwarder.remove_frame_handler(self._capture_frame)
            self._forwarder = None

    async def close(self) -> None:
        await self.stop_processing()
        logger.info("OCRProcessor closed (frames captured: %d)", self._frame_count)
