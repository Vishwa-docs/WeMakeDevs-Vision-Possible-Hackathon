"""Quick import test for Day 1 + Day 2 dependencies."""
import sys

errors = []

# --- Day 1 core ---
try:
    from vision_agents.core import Agent, AgentLauncher, Runner, User
    from vision_agents.plugins import gemini, getstream
    print("OK: Vision Agents SDK core imports")
except ImportError as e:
    errors.append(f"Vision Agents SDK: {e}")
    print(f"FAIL: {e}")

try:
    import cv2
    print(f"OK: OpenCV {cv2.__version__}")
except ImportError as e:
    errors.append(f"OpenCV: {e}")
    print(f"FAIL: {e}")

try:
    import aiosqlite
    print("OK: aiosqlite")
except ImportError as e:
    errors.append(f"aiosqlite: {e}")
    print(f"FAIL: {e}")

try:
    import httpx
    print(f"OK: httpx {httpx.__version__}")
except ImportError as e:
    errors.append(f"httpx: {e}")
    print(f"FAIL: {e}")

# --- Day 2 processors ---
try:
    from ultralytics import YOLO
    print(f"OK: ultralytics (YOLO) {YOLO.__module__}")
except ImportError as e:
    errors.append(f"ultralytics: {e}")
    print(f"FAIL: {e}")

try:
    from processors import (
        SignBridgeProcessor,
        SignDetectedEvent,
        GestureBufferEvent,
        SignTranslationEvent,
    )
    print(f"OK: SignBridgeProcessor ({SignBridgeProcessor.name})")
except ImportError as e:
    errors.append(f"SignBridgeProcessor: {e}")
    print(f"FAIL: {e}")

try:
    from processors import (
        GuideLensProcessor,
        ObjectDetectedEvent,
        HazardDetectedEvent,
        SceneSummaryEvent,
    )
    print(f"OK: GuideLensProcessor ({GuideLensProcessor.name})")
except ImportError as e:
    errors.append(f"GuideLensProcessor: {e}")
    print(f"FAIL: {e}")

try:
    from processors.signbridge_processor import GestureBuffer, GlossTranslator
    print("OK: GestureBuffer + GlossTranslator")
except ImportError as e:
    errors.append(f"GestureBuffer/GlossTranslator: {e}")
    print(f"FAIL: {e}")

try:
    from processors.guidelens_processor import BboxTracker
    print("OK: BboxTracker")
except ImportError as e:
    errors.append(f"BboxTracker: {e}")
    print(f"FAIL: {e}")

# --- Day 2 event system ---
try:
    from vision_agents.core.events import BaseEvent
    from vision_agents.core.processors import VideoProcessorPublisher
    from vision_agents.core.utils.video_forwarder import VideoForwarder
    from vision_agents.core.utils.video_track import QueuedVideoTrack
    print("OK: Vision Agents event system + video utils")
except ImportError as e:
    errors.append(f"Event system: {e}")
    print(f"FAIL: {e}")

print()
if errors:
    print(f"FAILED: {len(errors)} import(s) failed:")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)
else:
    print("All Day 1 + Day 2 imports verified!")
