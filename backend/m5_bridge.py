"""
M5StickC Plus Camera Bridge
============================
Bridges an RTSP/MJPEG camera stream (e.g. from M5StickC Plus) into the
Vision Agents framework via OpenCV → VideoForwarder.

Day 1: Stub with webcam fallback for development.
Day 7: Full M5StickC Plus RTSP integration.

Usage:
    python m5_bridge.py                         # Use default webcam (dev mode)
    python m5_bridge.py --url rtsp://IP:8554/   # Use RTSP stream
    python m5_bridge.py --url http://IP/stream   # Use MJPEG stream
"""

import argparse
import asyncio
import logging
import os
import time

import cv2
import numpy as np

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("m5_bridge")
logging.basicConfig(level=logging.INFO)


class CameraBridge:
    """Captures frames from an RTSP/MJPEG/webcam source."""

    def __init__(self, source: str | int = 0, target_fps: float = 10.0):
        """
        Args:
            source: RTSP URL, MJPEG URL, or integer webcam ID (0 = default cam).
            target_fps: Target capture rate (frames per second).
        """
        self.source = source
        self.target_fps = target_fps
        self._cap: cv2.VideoCapture | None = None
        self._running = False
        self._frame: np.ndarray | None = None
        self._frame_count = 0
        self._lock = asyncio.Lock()

    def open(self) -> bool:
        """Open the camera source."""
        logger.info("Opening camera source: %s", self.source)

        if isinstance(self.source, str) and self.source.startswith("rtsp://"):
            # RTSP: use TCP transport for reliability
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif isinstance(self.source, str) and self.source.startswith("http"):
            self._cap = cv2.VideoCapture(self.source)
        else:
            # Webcam
            self._cap = cv2.VideoCapture(int(self.source) if isinstance(self.source, str) else self.source)

        if not self._cap.isOpened():
            logger.error("Failed to open camera source: %s", self.source)
            return False

        # Read camera properties
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info("Camera opened: %dx%d @ %.1f FPS (target: %.1f FPS)", w, h, fps, self.target_fps)
        return True

    def read_frame(self) -> np.ndarray | None:
        """Read a single frame (blocking). Returns BGR ndarray or None."""
        if self._cap is None or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        self._frame_count += 1
        return frame

    async def capture_loop(self, on_frame=None):
        """
        Continuously capture frames at target FPS.
        Optionally call on_frame(frame) callback for each frame.
        """
        self._running = True
        interval = 1.0 / self.target_fps
        logger.info("Starting capture loop at %.1f FPS …", self.target_fps)

        loop = asyncio.get_event_loop()
        while self._running:
            t0 = time.monotonic()

            # Run blocking OpenCV read in thread pool
            frame = await loop.run_in_executor(None, self.read_frame)

            if frame is not None:
                async with self._lock:
                    self._frame = frame
                if on_frame:
                    await on_frame(frame)
            else:
                logger.warning("Frame capture failed — retrying …")
                await asyncio.sleep(1.0)
                continue

            # Throttle to target FPS
            elapsed = time.monotonic() - t0
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    @property
    def latest_frame(self) -> np.ndarray | None:
        """Get the most recently captured frame."""
        return self._frame

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def stop(self):
        """Stop the capture loop."""
        self._running = False

    def close(self):
        """Release camera resources."""
        self.stop()
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("Camera closed (total frames: %d)", self._frame_count)


class FrameStore:
    """Simple in-memory frame buffer with optional disk persistence."""

    def __init__(self, max_frames: int = 100, save_dir: str | None = None):
        self.max_frames = max_frames
        self.save_dir = save_dir
        self._frames: list[tuple[float, np.ndarray]] = []

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def add(self, frame: np.ndarray) -> None:
        ts = time.time()
        self._frames.append((ts, frame))

        # Evict oldest
        while len(self._frames) > self.max_frames:
            self._frames.pop(0)

        # Optionally save to disk
        if self.save_dir:
            path = os.path.join(self.save_dir, f"frame_{ts:.3f}.jpg")
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

    def get_recent(self, n: int = 10) -> list[tuple[float, np.ndarray]]:
        """Return the N most recent (timestamp, frame) pairs."""
        return self._frames[-n:]

    def clear(self):
        self._frames.clear()

    def __len__(self):
        return len(self._frames)


async def main():
    parser = argparse.ArgumentParser(description="M5StickC Plus Camera Bridge")
    parser.add_argument("--url", type=str, default=None, help="RTSP or MJPEG URL")
    parser.add_argument("--webcam", type=int, default=0, help="Webcam device ID")
    parser.add_argument("--fps", type=float, default=10.0, help="Target FPS")
    parser.add_argument("--save-frames", type=str, default=None, help="Directory to save frames")
    parser.add_argument("--show", action="store_true", help="Show live preview window")
    args = parser.parse_args()

    source = args.url or args.webcam
    bridge = CameraBridge(source=source, target_fps=args.fps)
    store = FrameStore(max_frames=50, save_dir=args.save_frames)

    if not bridge.open():
        logger.error("Cannot open camera — exiting.")
        return

    async def on_frame(frame: np.ndarray):
        store.add(frame)
        if bridge.frame_count % 30 == 0:
            logger.info("Captured %d frames (buffer: %d)", bridge.frame_count, len(store))

        if args.show:
            cv2.imshow("M5 Bridge", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                bridge.stop()

    try:
        await bridge.capture_loop(on_frame=on_frame)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
