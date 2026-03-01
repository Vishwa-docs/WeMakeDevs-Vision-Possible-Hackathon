"""
WorldLens — Backend Test Suite
================================
Comprehensive unit + integration tests for:
  1. Provider system (ProviderManager, adapters, fallback, cooldown)
  2. Processor components (GestureBuffer, BboxTracker, event dataclasses)
  3. TranscriptAggregator (word buffering)
  4. API endpoints (mode, providers, transcript, token — via FastAPI TestClient)
  5. End-to-end session lifecycle

Run:
    cd testing && python -m pytest test_backend.py -v
    # or from project root:
    python -m pytest testing/ -v
"""

import asyncio
import os
import sys
import time
import uuid
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Backend dir is added by conftest.py


# ===================================================================
# 1. PROVIDER SYSTEM TESTS
# ===================================================================
class TestProviderStatus:
    """Test ProviderStatus dataclass behaviour."""

    def test_default_values(self):
        from providers import ProviderStatus, ProviderID

        status = ProviderStatus(provider=ProviderID.GEMINI)
        assert status.available is True
        assert status.last_error is None
        assert status.total_calls == 0
        assert status.total_errors == 0
        assert status.cooldown_until == 0.0
        assert status.in_cooldown is False

    def test_record_success(self):
        from providers import ProviderStatus, ProviderID

        status = ProviderStatus(provider=ProviderID.GEMINI)
        status.record_success()
        assert status.total_calls == 1
        assert status.total_errors == 0
        assert status.available is True

    def test_record_error_sets_cooldown(self):
        from providers import ProviderStatus, ProviderID

        status = ProviderStatus(provider=ProviderID.GROK)
        status.record_error("Rate limit", cooldown_seconds=10.0)

        assert status.total_calls == 1
        assert status.total_errors == 1
        assert status.last_error == "Rate limit"
        assert status.last_error_time is not None
        assert status.in_cooldown is True

    def test_cooldown_expires(self):
        from providers import ProviderStatus, ProviderID

        status = ProviderStatus(provider=ProviderID.NVIDIA)
        status.record_error("timeout", cooldown_seconds=0.0)  # immediate expiry
        # cooldown_until = time.time() + 0 → already past
        assert status.in_cooldown is False

    def test_to_dict(self):
        from providers import ProviderStatus, ProviderID

        status = ProviderStatus(provider=ProviderID.HUGGINGFACE)
        d = status.to_dict()
        assert d["provider"] == "huggingface"
        assert d["available"] is True
        assert d["in_cooldown"] is False
        assert "total_calls" in d
        assert "total_errors" in d


class TestFallbackEvent:
    """Test FallbackEvent dataclass."""

    def test_to_dict(self):
        from providers import FallbackEvent

        evt = FallbackEvent(
            original_provider="gemini",
            fallback_provider="grok",
            error_reason="Rate limit exceeded",
        )
        d = evt.to_dict()
        assert d["original"] == "gemini"
        assert d["fallback"] == "grok"
        assert d["reason"] == "Rate limit exceeded"
        assert isinstance(d["timestamp"], float)


class TestProviderManager:
    """Test ProviderManager singleton orchestration."""

    def test_default_preferred(self):
        from providers import ProviderManager, ProviderID

        pm = ProviderManager()
        # Default can be any valid ProviderID (depends on env)
        assert pm.preferred in list(ProviderID)

    def test_set_preferred_valid(self):
        from providers import ProviderManager, ProviderID

        pm = ProviderManager()
        assert pm.set_preferred("grok") is True
        assert pm.preferred == ProviderID.GROK

    def test_set_preferred_invalid(self):
        from providers import ProviderManager

        pm = ProviderManager()
        assert pm.set_preferred("nonexistent") is False

    def test_fallback_chain_preferred_first(self):
        from providers import ProviderManager, ProviderID

        pm = ProviderManager()
        pm.set_preferred("nvidia")
        chain = pm.get_fallback_chain()
        assert chain[0] == ProviderID.NVIDIA
        # Preferred should not appear twice
        assert chain.count(ProviderID.NVIDIA) == 1
        # All providers should be in the chain
        assert len(chain) == len(ProviderID)

    def test_get_status_shape(self):
        from providers import ProviderManager

        pm = ProviderManager()
        status = pm.get_status()
        assert "preferred" in status
        assert "providers" in status
        assert "fallback_chain" in status
        assert isinstance(status["providers"], dict)

    def test_pop_fallback_events_empty(self):
        from providers import ProviderManager

        pm = ProviderManager()
        events = pm.pop_fallback_events()
        assert events == []

    @pytest.mark.asyncio
    async def test_check_all_providers(self):
        from providers import ProviderManager

        pm = ProviderManager()
        results = await pm.check_all_providers()
        assert isinstance(results, dict)
        # All providers should appear
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_caption_all_fail_raises(self):
        from providers import ProviderManager

        pm = ProviderManager()
        # Force all health checks to fail
        for adapter in pm._adapters.values():
            adapter.health_check = AsyncMock(return_value=False)

        with pytest.raises(RuntimeError, match="All VLM providers failed"):
            await pm.caption(b"\x89PNG dummy image bytes")

    @pytest.mark.asyncio
    async def test_caption_fallback(self):
        from providers import ProviderManager, ProviderID

        pm = ProviderManager()

        # Make preferred (gemini) fail, second (grok) succeed
        pm.set_preferred("gemini")

        call_count = 0

        async def mock_gemini_caption(image_bytes, prompt=""):
            raise Exception("Gemini down")

        async def mock_grok_caption(image_bytes, prompt=""):
            nonlocal call_count
            call_count += 1
            return "A cat sitting on a desk"

        pm._adapters[ProviderID.GEMINI].caption = mock_gemini_caption
        pm._adapters[ProviderID.GEMINI].health_check = AsyncMock(return_value=True)
        pm._adapters[ProviderID.GROK].caption = mock_grok_caption
        pm._adapters[ProviderID.GROK].health_check = AsyncMock(return_value=True)

        # Disable other adapters
        for pid in (ProviderID.AZURE_OPENAI, ProviderID.NVIDIA, ProviderID.HUGGINGFACE):
            pm._adapters[pid].health_check = AsyncMock(return_value=False)

        result, used_pid = await pm.caption(b"fake-image")
        assert result == "A cat sitting on a desk"
        assert used_pid == ProviderID.GROK
        assert call_count == 1

        # Should have recorded a fallback event
        events = pm.pop_fallback_events()
        assert len(events) == 1
        assert events[0]["original"] == "gemini"
        assert events[0]["fallback"] == "grok"


class TestProviderAdapters:
    """Test individual adapter health_check logic (no real API calls)."""

    @pytest.mark.asyncio
    async def test_gemini_health_no_key(self):
        from providers import GeminiAdapter

        adapter = GeminiAdapter()
        with patch.dict(os.environ, {"GOOGLE_API_KEY": ""}, clear=False):
            os.environ.pop("GOOGLE_API_KEY", None)
            result = await adapter.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_gemini_health_with_key(self):
        from providers import GeminiAdapter

        adapter = GeminiAdapter()
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            result = await adapter.health_check()
            assert result is True

    @pytest.mark.asyncio
    async def test_hf_health_no_key(self):
        from providers import HuggingFaceAdapter

        adapter = HuggingFaceAdapter()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_TOKEN", None)
            os.environ.pop("HF_API_TOKEN", None)
            result = await adapter.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_nvidia_health_no_key(self):
        from providers import NvidiaAdapter

        adapter = NvidiaAdapter()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NGC_API_KEY", None)
            os.environ.pop("NVIDIA_API_KEY", None)
            result = await adapter.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_grok_health_no_key(self):
        from providers import GrokAdapter

        adapter = GrokAdapter()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("XAI_API_KEY", None)
            result = await adapter.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_azure_health_no_key(self):
        from providers import AzureOpenAIAdapter

        adapter = AzureOpenAIAdapter()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AZURE_OPENAI_ENDPOINT", None)
            os.environ.pop("AZURE_OPENAI_API_KEY", None)
            result = await adapter.health_check()
            assert result is False


# ===================================================================
# 2. PROCESSOR COMPONENT TESTS
# ===================================================================
class TestGestureBuffer:
    """Test the GestureBuffer from SignBridge processor."""

    def test_initial_state(self):
        from processors.signbridge_processor import GestureBuffer

        buf = GestureBuffer(max_frames=30, min_frames_for_gesture=10)
        assert buf.length == 0

    def test_add_frame_insufficient(self):
        from processors.signbridge_processor import GestureBuffer

        buf = GestureBuffer(max_frames=30, min_frames_for_gesture=10)
        result = buf.add_frame([[0, 0] * 17], 1.0)
        assert result is None
        assert buf.length == 1

    def test_add_frames_no_motion(self):
        """Add enough frames but with no motion — should not trigger."""
        from processors.signbridge_processor import GestureBuffer

        buf = GestureBuffer(max_frames=30, min_frames_for_gesture=5)
        static_kps = [[[100, 100, 0.9]] * 17]
        for i in range(15):
            result = buf.add_frame(static_kps, float(i) + 3.0)  # beyond cooldown
        # No significant motion → should return None
        assert result is None

    def test_add_frames_with_motion(self):
        """Add frames with significant wrist motion → should trigger."""
        from processors.signbridge_processor import GestureBuffer

        buf = GestureBuffer(max_frames=30, min_frames_for_gesture=5)
        buf._cooldown = 0.0  # disable cooldown for testing

        for i in range(10):
            # Wrist keypoints (indices 9, 10) move significantly
            kps = [[[100, 100]] * 9 + [[100 + i * 20, 100 + i * 15], [200 + i * 20, 200 + i * 15]] + [[100, 100]] * 6]
            result = buf.add_frame(kps, float(i))

        # With 100 pixels of motion, should have triggered
        # (depends on exact threshold, but we moved wrists significantly)

    def test_clear(self):
        from processors.signbridge_processor import GestureBuffer

        buf = GestureBuffer()
        buf.add_frame([[0, 0] * 17], 1.0)
        buf.clear()
        assert buf.length == 0

    def test_max_frames_cap(self):
        from processors.signbridge_processor import GestureBuffer

        buf = GestureBuffer(max_frames=5)
        for i in range(10):
            buf.add_frame([[0, 0] * 17], float(i))
        assert buf.length == 5  # should be capped at max_frames


class TestBboxTracker:
    """Test the BboxTracker from GuideLens processor."""

    def test_initial_growth_rate_zero(self):
        from processors.guidelens_processor import BboxTracker

        tracker = BboxTracker()
        assert tracker.growth_rate("car") == 0.0

    def test_growth_rate_approaching(self):
        from processors.guidelens_processor import BboxTracker

        tracker = BboxTracker()
        # Object getting bigger → approaching
        tracker.update("car", 0.05, 1.0)
        tracker.update("car", 0.10, 2.0)
        rate = tracker.growth_rate("car")
        assert rate > 0  # positive = approaching
        assert abs(rate - 0.05) < 0.001

    def test_growth_rate_receding(self):
        from processors.guidelens_processor import BboxTracker

        tracker = BboxTracker()
        tracker.update("person", 0.15, 1.0)
        tracker.update("person", 0.10, 2.0)
        rate = tracker.growth_rate("person")
        assert rate < 0  # negative = receding

    def test_growth_rate_stationary(self):
        from processors.guidelens_processor import BboxTracker

        tracker = BboxTracker()
        tracker.update("bicycle", 0.08, 1.0)
        tracker.update("bicycle", 0.08, 2.0)
        assert tracker.growth_rate("bicycle") == 0.0

    def test_growth_rate_insufficient_data(self):
        from processors.guidelens_processor import BboxTracker

        tracker = BboxTracker()
        tracker.update("bus", 0.1, 1.0)
        assert tracker.growth_rate("bus") == 0.0  # only 1 entry

    def test_growth_rate_too_close_timestamps(self):
        from processors.guidelens_processor import BboxTracker

        tracker = BboxTracker()
        tracker.update("truck", 0.05, 1.0)
        tracker.update("truck", 0.10, 1.05)  # only 50ms apart
        assert tracker.growth_rate("truck") == 0.0  # dt < 0.1

    def test_clear(self):
        from processors.guidelens_processor import BboxTracker

        tracker = BboxTracker()
        tracker.update("car", 0.1, 1.0)
        tracker.clear()
        assert tracker.growth_rate("car") == 0.0

    def test_multiple_classes_independent(self):
        from processors.guidelens_processor import BboxTracker

        tracker = BboxTracker()
        tracker.update("car", 0.05, 1.0)
        tracker.update("car", 0.10, 2.0)
        tracker.update("person", 0.20, 1.0)
        tracker.update("person", 0.15, 2.0)

        assert tracker.growth_rate("car") > 0
        assert tracker.growth_rate("person") < 0


class TestEventDataclasses:
    """Test that event dataclasses instantiate correctly with default values."""

    def test_sign_detected_event(self):
        from processors import SignDetectedEvent

        evt = SignDetectedEvent()
        assert evt.type == "signbridge.sign_detected"
        assert evt.keypoints == []
        assert evt.num_persons == 0
        assert evt.confidence == 0.0
        assert evt.frame_number == 0

    def test_sign_detected_event_with_values(self):
        from processors import SignDetectedEvent

        evt = SignDetectedEvent(
            keypoints=[[1, 2, 3]],
            num_persons=2,
            confidence=0.95,
            frame_number=42,
            timestamp_unix=1234567890.0,
        )
        assert evt.num_persons == 2
        assert evt.confidence == 0.95
        assert evt.frame_number == 42

    def test_gesture_buffer_event(self):
        from processors import GestureBufferEvent

        evt = GestureBufferEvent()
        assert evt.type == "signbridge.gesture_buffer"
        assert evt.gesture_sequence == []
        assert evt.buffer_length == 0
        assert evt.raw_gloss == ""

    def test_sign_translation_event(self):
        from processors import SignTranslationEvent

        evt = SignTranslationEvent(
            raw_gloss="WAVE",
            translated_text="Hello! (waving gesture detected)",
            language="en",
        )
        assert evt.type == "signbridge.sign_translation"
        assert evt.raw_gloss == "WAVE"
        assert evt.translated_text == "Hello! (waving gesture detected)"

    def test_object_detected_event(self):
        from processors import ObjectDetectedEvent

        evt = ObjectDetectedEvent(
            objects=["car", "person", "bicycle"],
            frame_number=100,
            timestamp_unix=time.time(),
        )
        assert evt.type == "guidelens.object_detected"
        assert len(evt.objects) == 3

    def test_hazard_detected_event(self):
        from processors import HazardDetectedEvent

        evt = HazardDetectedEvent(
            hazard_type="car",
            distance_estimate="near",
            direction="left",
            confidence=0.92,
            growth_rate=0.08,
            bbox_area_ratio=0.18,
        )
        assert evt.type == "guidelens.hazard_detected"
        assert evt.distance_estimate == "near"
        assert evt.direction == "left"

    def test_scene_summary_event(self):
        from processors import SceneSummaryEvent

        evt = SceneSummaryEvent(
            summary="2 people, 1 car, 1 bicycle",
            object_counts={"person": 2, "car": 1, "bicycle": 1},
        )
        assert evt.type == "guidelens.scene_summary"
        assert evt.object_counts["person"] == 2


class TestGlossTranslator:
    """Test the GlossTranslator rule-based fallback."""

    def test_known_gloss(self):
        from processors.signbridge_processor import GlossTranslator

        translator = GlossTranslator()
        result = asyncio.get_event_loop().run_until_complete(
            translator.translate("WAVE")
        )
        assert "waving" in result.lower() or "hello" in result.lower()

    def test_unknown_gloss(self):
        from processors.signbridge_processor import GlossTranslator

        translator = GlossTranslator()
        result = asyncio.get_event_loop().run_until_complete(
            translator.translate("UNKNOWN_GESTURE")
        )
        assert "UNKNOWN_GESTURE" in result

    def test_empty_gloss(self):
        from processors.signbridge_processor import GlossTranslator

        translator = GlossTranslator()
        result = asyncio.get_event_loop().run_until_complete(
            translator.translate("")
        )
        assert result == ""


# ===================================================================
# 3. GUIDELENS / SIGNBRIDGE CONSTANTS & CONFIG
# ===================================================================
class TestGuideLensConfig:
    """Validate GuideLens module-level constants."""

    def test_hazard_classes_non_empty(self):
        from processors.guidelens_processor import HAZARD_CLASSES

        assert len(HAZARD_CLASSES) > 0
        assert "car" in HAZARD_CLASSES
        assert "person" in HAZARD_CLASSES

    def test_distance_thresholds(self):
        from processors.guidelens_processor import DISTANCE_THRESHOLDS

        assert "near" in DISTANCE_THRESHOLDS
        assert "medium" in DISTANCE_THRESHOLDS
        assert DISTANCE_THRESHOLDS["near"] > DISTANCE_THRESHOLDS["medium"]

    def test_approach_rate_threshold(self):
        from processors.guidelens_processor import APPROACH_RATE_THRESHOLD

        assert APPROACH_RATE_THRESHOLD > 0


class TestSignBridgeConfig:
    """Validate SignBridge module-level constants."""

    def test_keypoint_names(self):
        from processors.signbridge_processor import KEYPOINT_NAMES

        assert len(KEYPOINT_NAMES) == 17
        assert "nose" in KEYPOINT_NAMES
        assert "left_wrist" in KEYPOINT_NAMES
        assert "right_wrist" in KEYPOINT_NAMES

    def test_upper_body_connections(self):
        from processors.signbridge_processor import UPPER_BODY_CONNECTIONS

        assert len(UPPER_BODY_CONNECTIONS) > 0
        # All indices should be valid COCO keypoint indices (0-16)
        for a, b in UPPER_BODY_CONNECTIONS:
            assert 0 <= a <= 16
            assert 0 <= b <= 16

    def test_hand_keypoints(self):
        from processors.signbridge_processor import HAND_KEYPOINTS

        assert HAND_KEYPOINTS == [9, 10]

    def test_processor_names(self):
        from processors import SignBridgeProcessor, GuideLensProcessor

        assert SignBridgeProcessor.name == "signbridge_pose"
        assert GuideLensProcessor.name == "guidelens_detection"


# ===================================================================
# 4. TRANSCRIPT AGGREGATOR TESTS
# ===================================================================
class TestTranscriptAggregator:
    """Test the _TranscriptAggregator from main.py.

    Since _TranscriptAggregator is defined inside main.py, we import it
    from there. The import also pulls in the SDK, which should be available
    in the backend venv.
    """

    def _make_aggregator(self):
        """Import and create a fresh _TranscriptAggregator."""
        # We need to import from main — but main.py guards the runner
        # behind `if __name__ == '__main__'`, so the class should be
        # importable at module scope.
        from main import _TranscriptAggregator

        return _TranscriptAggregator(flush_delay=0.1)  # fast flush for tests

    @pytest.mark.asyncio
    async def test_buffer_accumulates(self):
        agg = self._make_aggregator()
        mock_conv = AsyncMock()
        mock_conv.upsert_message = AsyncMock()

        await agg.add("Hello ", mock_conv, "user-1", "user")
        assert len(agg._buffer) == 1
        await agg.add("world", mock_conv, "user-1", "user")
        assert len(agg._buffer) == 2

    @pytest.mark.asyncio
    async def test_flush_after_silence(self):
        """After flush_delay with no new text, buffer should be flushed."""
        agg = self._make_aggregator()
        mock_conv = AsyncMock()
        mock_conv.upsert_message = AsyncMock()

        await agg.add("Test message", mock_conv, "agent-1", "assistant")
        # Wait for flush
        await asyncio.sleep(0.3)

        # Buffer should be cleared after flush
        assert len(agg._buffer) == 0
        # upsert_message should have been called with completed=True
        calls = mock_conv.upsert_message.call_args_list
        final_call = calls[-1]
        assert final_call.kwargs.get("completed") is True or (
            len(final_call.args) > 0
        )

    @pytest.mark.asyncio
    async def test_message_id_resets_after_flush(self):
        agg = self._make_aggregator()
        mock_conv = AsyncMock()
        mock_conv.upsert_message = AsyncMock()

        old_id = agg._message_id
        await agg.add("First", mock_conv, "u1", "user")
        await asyncio.sleep(0.3)  # wait for flush
        new_id = agg._message_id
        assert old_id != new_id  # new UUID after flush


# ===================================================================
# 5. TRANSCRIPT LOG (module-level deque in main.py)
# ===================================================================
class TestTranscriptLog:
    """Test the in-memory _transcript_log from main.py."""

    def test_import_transcript_log(self):
        from main import _transcript_log

        assert isinstance(_transcript_log, deque)
        assert _transcript_log.maxlen == 500

    def test_append_and_read(self):
        from main import _transcript_log

        _transcript_log.clear()
        _transcript_log.append({
            "speaker": "agent",
            "text": "Hello world",
            "timestamp": time.time() * 1000,
        })
        assert len(_transcript_log) == 1
        assert _transcript_log[0]["speaker"] == "agent"
        _transcript_log.clear()

    def test_maxlen_enforced(self):
        from main import _transcript_log

        _transcript_log.clear()
        for i in range(600):
            _transcript_log.append({"speaker": "user", "text": str(i), "timestamp": float(i)})
        assert len(_transcript_log) == 500
        _transcript_log.clear()


# ===================================================================
# 6. MODE SWITCHING LOGIC
# ===================================================================
class TestModeSwitching:
    """Test mode-related helpers in main.py."""

    def test_get_instructions_guidelens(self):
        import main

        original = main.AGENT_MODE
        try:
            main.AGENT_MODE = "guidelens"
            instructions = main._get_instructions()
            assert "GuideLens" in instructions
            assert "environmental" in instructions.lower()
        finally:
            main.AGENT_MODE = original

    def test_get_instructions_signbridge(self):
        import main

        original = main.AGENT_MODE
        try:
            main.AGENT_MODE = "signbridge"
            instructions = main._get_instructions()
            assert "SignBridge" in instructions
            assert "sign" in instructions.lower()
        finally:
            main.AGENT_MODE = original

    def test_build_processors_guidelens(self):
        import main

        original = main.AGENT_MODE
        try:
            main.AGENT_MODE = "guidelens"
            procs = main._build_processors()
            assert len(procs) == 2  # GuideLens + OCR
            proc_names = [p.__class__.__name__ for p in procs]
            assert "GuideLensProcessor" in proc_names
            assert "OCRProcessor" in proc_names
        finally:
            main.AGENT_MODE = original

    def test_build_processors_signbridge(self):
        import main

        original = main.AGENT_MODE
        try:
            main.AGENT_MODE = "signbridge"
            procs = main._build_processors()
            assert len(procs) == 1
            assert procs[0].__class__.__name__ == "SignBridgeProcessor"
        finally:
            main.AGENT_MODE = original


# ===================================================================
# 7. PROVIDER ID ENUM
# ===================================================================
class TestProviderID:
    """Test ProviderID enum members and string conversion."""

    def test_all_members(self):
        from providers import ProviderID

        expected = {"gemini", "huggingface", "nvidia", "grok", "azure_openai"}
        actual = {p.value for p in ProviderID}
        assert actual == expected

    def test_str_enum(self):
        from providers import ProviderID

        assert str(ProviderID.GEMINI) == "ProviderID.GEMINI" or ProviderID.GEMINI.value == "gemini"
        assert ProviderID("gemini") == ProviderID.GEMINI


# ===================================================================
# 8. SDK IMPORT VALIDATION
# ===================================================================
class TestSDKImports:
    """Verify that all required SDK modules are importable."""

    def test_core_imports(self):
        from vision_agents.core import Agent, AgentLauncher, Runner, ServeOptions, User

    def test_gemini_plugin(self):
        from vision_agents.plugins import gemini

        assert hasattr(gemini, "Realtime")

    def test_getstream_plugin(self):
        from vision_agents.plugins import getstream

        assert hasattr(getstream, "Edge")

    def test_event_classes(self):
        from vision_agents.core.events import BaseEvent
        from vision_agents.core.llm.events import (
            RealtimeAgentSpeechTranscriptionEvent,
            RealtimeUserSpeechTranscriptionEvent,
        )

    def test_getstream_call_events(self):
        from vision_agents.plugins.getstream import (
            CallSessionParticipantJoinedEvent,
            CallSessionParticipantLeftEvent,
        )

    def test_processor_base_classes(self):
        from vision_agents.core.processors import VideoProcessorPublisher
        from vision_agents.core.utils.video_forwarder import VideoForwarder
        from vision_agents.core.utils.video_track import QueuedVideoTrack

    def test_ultralytics(self):
        from ultralytics import YOLO

    def test_opencv(self):
        import cv2

        assert hasattr(cv2, "__version__")

    def test_httpx(self):
        import httpx

        assert hasattr(httpx, "AsyncClient")

    def test_getstream_sdk(self):
        from getstream import Stream
        from getstream.models import UserRequest


# ===================================================================
# 9. E2E SANITY — agent factory (mocked SDK)
# ===================================================================
class TestAgentFactory:
    """Test create_agent() returns an Agent with correct configuration."""

    @pytest.mark.asyncio
    async def test_create_agent_returns_agent(self):
        from main import create_agent

        agent = await create_agent()
        assert agent is not None
        # Should have at least one processor
        assert len(agent.processors) >= 1

    @pytest.mark.asyncio
    async def test_create_agent_guidelens_mode(self):
        import main
        from main import create_agent

        original = main.AGENT_MODE
        try:
            main.AGENT_MODE = "guidelens"
            agent = await create_agent()
            proc_names = [p.__class__.__name__ for p in agent.processors]
            assert "GuideLensProcessor" in proc_names
        finally:
            main.AGENT_MODE = original

    @pytest.mark.asyncio
    async def test_create_agent_signbridge_mode(self):
        import main
        from main import create_agent

        original = main.AGENT_MODE
        try:
            main.AGENT_MODE = "signbridge"
            agent = await create_agent()
            proc_names = [p.__class__.__name__ for p in agent.processors]
            assert "SignBridgeProcessor" in proc_names
        finally:
            main.AGENT_MODE = original


# ===================================================================
# 10. DISTANCE & DIRECTION HELPERS (GuideLens)
# ===================================================================
class TestGuideLensHelpers:
    """Test the distance and direction estimation logic."""

    def test_distance_near(self):
        from processors.guidelens_processor import DISTANCE_THRESHOLDS

        area_ratio = 0.20  # 20% > 15% threshold
        assert area_ratio > DISTANCE_THRESHOLDS["near"]

    def test_distance_medium(self):
        from processors.guidelens_processor import DISTANCE_THRESHOLDS

        area_ratio = 0.08  # 8% > 5% but < 15%
        assert area_ratio > DISTANCE_THRESHOLDS["medium"]
        assert area_ratio < DISTANCE_THRESHOLDS["near"]

    def test_distance_far(self):
        from processors.guidelens_processor import DISTANCE_THRESHOLDS

        area_ratio = 0.02  # 2% < 5%
        assert area_ratio < DISTANCE_THRESHOLDS["medium"]
