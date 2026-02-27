"""Day 2 comprehensive validation test."""
import sys

print("=== DAY 2 VALIDATION ===\n")

# 1. Import test
from processors import (
    SignBridgeProcessor, SignDetectedEvent, GestureBufferEvent, SignTranslationEvent,
    GuideLensProcessor, ObjectDetectedEvent, HazardDetectedEvent, SceneSummaryEvent,
)
print("1. All processor imports OK")

# 2. Instantiate processors
sb = SignBridgeProcessor(fps=10, conf_threshold=0.5, model_path="yolo11n-pose.pt")
print(f"2. SignBridgeProcessor created: name={sb.name}")

gl = GuideLensProcessor(fps=5, conf_threshold=0.4, model_path="yolo11n.pt")
print(f"3. GuideLensProcessor created: name={gl.name}")

# 3. Test event creation
e1 = SignDetectedEvent(num_persons=2, confidence=0.95, frame_number=42, timestamp_unix=0.0)
print(f"4. SignDetectedEvent: type={e1.type}, persons={e1.num_persons}")

e2 = HazardDetectedEvent(hazard_type="car", distance_estimate="near", direction="left")
print(f"5. HazardDetectedEvent: type={e2.type}, hazard={e2.hazard_type}")

e3 = SceneSummaryEvent(summary="3 persons, 1 car")
print(f"6. SceneSummaryEvent: type={e3.type}, summary={e3.summary}")

e4 = GestureBufferEvent(raw_gloss="WAVE", buffer_length=20)
print(f"7. GestureBufferEvent: type={e4.type}, gloss={e4.raw_gloss}")

e5 = SignTranslationEvent(raw_gloss="WAVE", translated_text="Hello!")
print(f"8. SignTranslationEvent: type={e5.type}, translation={e5.translated_text}")

# 4. Test gesture buffer
from processors.signbridge_processor import GestureBuffer
gb = GestureBuffer(max_frames=10, min_frames_for_gesture=3)
print(f"9. GestureBuffer created: max={gb.max_frames}")

# 5. Test bbox tracker
from processors.guidelens_processor import BboxTracker
bt = BboxTracker()
bt.update("person", 0.05, 1.0)
bt.update("person", 0.08, 2.0)
rate = bt.growth_rate("person")
print(f"10. BboxTracker: person growth_rate={rate:.3f}/s")

# 6. Verify YOLO models load
print(f"11. SignBridge model loaded: {sb._model is not None}")
print(f"12. GuideLens model loaded: {gl._model is not None}")

# 7. Main module import
import main
print(f"13. main.py import OK, mode={main.AGENT_MODE}")

print("\n=== ALL DAY 2 VALIDATIONS PASSED ===")
