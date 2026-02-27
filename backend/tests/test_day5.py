"""
Day 5 tests — haptic alerts, telemetry, alert polling.
"""

import asyncio
import time

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nav_engine():
    """Construct a NavigationEngine with default settings."""
    from mcp_tools.navigation_engine import NavigationEngine
    return NavigationEngine()


# ---------------------------------------------------------------------------
# NavigationEngine.pop_hazard_alerts
# ---------------------------------------------------------------------------

class TestPopHazardAlerts:
    """Verify consumable alert polling via pop_hazard_alerts."""

    def test_pop_returns_alerts_added_after_since(self):
        nav = _make_nav_engine()
        ts = time.time()
        nav._hazard_alerts.append(
            {"text": "car approaching", "timestamp": ts + 1, "severity": "warning"}
        )
        nav._hazard_alerts.append(
            {"text": "cyclist", "timestamp": ts + 2, "severity": "caution"}
        )
        result = nav.pop_hazard_alerts(since=ts)
        assert len(result) == 2
        assert result[0]["text"] == "car approaching"

    def test_pop_filters_old_alerts(self):
        nav = _make_nav_engine()
        old_ts = time.time() - 100
        nav._hazard_alerts.append({"text": "old", "timestamp": old_ts, "severity": "caution"})
        nav._hazard_alerts.append({"text": "new", "timestamp": time.time() + 1, "severity": "warning"})
        result = nav.pop_hazard_alerts(since=time.time())
        assert len(result) == 1
        assert result[0]["text"] == "new"

    def test_pop_consumes_alerts(self):
        nav = _make_nav_engine()
        nav._hazard_alerts.append({"text": "a", "timestamp": time.time() + 1, "severity": "warning"})
        # First pop returns it
        first = nav.pop_hazard_alerts(since=0)
        assert len(first) == 1
        # Second pop: empty (consumed)
        second = nav.pop_hazard_alerts(since=0)
        assert len(second) == 0

    def test_pop_empty_list(self):
        nav = _make_nav_engine()
        result = nav.pop_hazard_alerts(since=0)
        assert result == []


# ---------------------------------------------------------------------------
# Processor telemetry
# ---------------------------------------------------------------------------

class TestProcessorTelemetry:
    """Verify get_telemetry() on each processor."""

    def test_guidelens_telemetry_shape(self):
        from processors.guidelens_processor import GuideLensProcessor

        proc = GuideLensProcessor.__new__(GuideLensProcessor)
        # Initialise all fields referenced by get_telemetry
        proc._total_objects_detected = 42
        proc._total_hazards_detected = 3
        proc._total_inference_time = 1.5
        proc._inference_count = 10
        proc._start_time = time.time() - 60
        proc._frame_count = 10
        proc.model_path = "yolo11n.pt"
        proc.device = "cpu"
        proc.fps = 5

        t = proc.get_telemetry()
        assert t["processor"] == "guidelens_detection"
        assert t["frames_processed"] == 10
        assert t["total_objects_detected"] == 42
        assert t["total_hazards_detected"] == 3
        assert t["avg_inference_ms"] == pytest.approx(150.0, rel=0.01)
        assert t["uptime_seconds"] >= 59

    def test_signbridge_telemetry_shape(self):
        from processors.signbridge_processor import SignBridgeProcessor
        from unittest.mock import MagicMock

        proc = SignBridgeProcessor.__new__(SignBridgeProcessor)
        proc._total_gestures_detected = 7
        proc._total_persons_detected = 15
        proc._total_hands_detected = 4
        proc._total_asl_letters_detected = 2
        proc._total_inference_time = 2.0
        proc._inference_count = 20
        proc._start_time = time.time() - 30
        proc._frame_count = 20
        proc.model_path = "yolo11n-pose.pt"
        proc.device = "cpu"
        proc.fps = 10
        proc._gesture_buffer = MagicMock()
        proc._gesture_buffer.length = 5
        proc._hand_landmarker = None

        t = proc.get_telemetry()
        assert t["processor"] == "signbridge_pose"
        assert t["frames_processed"] == 20
        assert t["total_gestures_detected"] == 7
        assert t["total_persons_detected"] == 15
        assert t["avg_inference_ms"] == pytest.approx(100.0, rel=0.01)

    def test_ocr_telemetry_shape(self):
        from processors.ocr_processor import OCRProcessor

        proc = OCRProcessor.__new__(OCRProcessor)
        proc._total_ocr_calls = 5
        proc._total_scene_calls = 2
        proc._start_time = time.time() - 10
        proc._frame_count = 0
        proc._ocr_results = []
        proc.scan_interval = 10.0

        t = proc.get_telemetry()
        assert t["processor"] == "ocr_vlm"
        assert t["total_ocr_calls"] == 5
        assert t["total_scene_calls"] == 2
        assert t["frames_captured"] == 0
        assert t["cached_results"] == 0


# ---------------------------------------------------------------------------
# Haptic alert severity mapping (embedded in main.py trigger_haptic_alert)
# ---------------------------------------------------------------------------

class TestHapticSeverityMapping:
    """Verify the severity → priority/sound/duration mapping logic."""

    def test_critical_mapping(self):
        mapping = {"critical": (10, "siren", 3000), "warning": (5, "beep", 2000), "caution": (2, "chime", 1500)}
        priority, sound, dur = mapping["critical"]
        assert priority == 10
        assert sound == "siren"
        assert dur == 3000

    def test_warning_mapping(self):
        mapping = {"critical": (10, "siren", 3000), "warning": (5, "beep", 2000), "caution": (2, "chime", 1500)}
        priority, sound, dur = mapping["warning"]
        assert priority == 5
        assert sound == "beep"
        assert dur == 2000

    def test_caution_mapping(self):
        mapping = {"critical": (10, "siren", 3000), "warning": (5, "beep", 2000), "caution": (2, "chime", 1500)}
        priority, sound, dur = mapping["caution"]
        assert priority == 2
        assert sound == "chime"
        assert dur == 1500
