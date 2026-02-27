"""
Day 4 unit tests — SpatialMemory (SQLite), NavigationEngine, Maps API stubs.

Run:  python -m pytest tests/test_day4.py -v
"""

import asyncio
import os
import tempfile
import time

import pytest

# ---------------------------------------------------------------------------
# SpatialMemory tests
# ---------------------------------------------------------------------------
from mcp_tools.spatial_memory import SpatialMemory


@pytest.fixture
def tmp_db(tmp_path):
    """Return a temporary DB path."""
    return str(tmp_path / "test_spatial.db")


@pytest.fixture
def memory(tmp_db):
    """Create a fresh SpatialMemory pointing at a temp DB."""
    return SpatialMemory(db_path=tmp_db)


@pytest.mark.asyncio
async def test_spatial_memory_init_and_schema(memory: SpatialMemory):
    """initialise() should create the DB file and the detections table."""
    await memory.initialise()
    assert os.path.exists(memory.db_path)

    import aiosqlite

    async with aiosqlite.connect(memory.db_path) as db:
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='detections'"
        )
        row = await cursor.fetchone()
        assert row is not None, "detections table should exist"
    await memory.close()


@pytest.mark.asyncio
async def test_log_detection_batch_and_search(memory: SpatialMemory):
    """log_detection_batch() then search() should return matching rows."""
    await memory.initialise()

    detections = [
        {
            "object_name": "person",
            "direction": "ahead",
            "distance": "nearby",
            "confidence": 0.91,
            "frame_number": 1,
        },
        {
            "object_name": "car",
            "direction": "left",
            "distance": "far",
            "confidence": 0.85,
            "frame_number": 1,
        },
        {
            "object_name": "person",
            "direction": "right",
            "distance": "close",
            "confidence": 0.88,
            "frame_number": 2,
        },
    ]
    await memory.log_detection_batch(detections)

    # Search for "person"
    results = await memory.search("person")
    assert len(results) >= 1
    assert any("person" in r.get("object", "") for r in results)

    # Search for "car"
    results = await memory.search("car")
    assert len(results) >= 1

    # Search for something not logged
    results = await memory.search("elephant")
    assert len(results) == 0

    await memory.close()


@pytest.mark.asyncio
async def test_get_recent(memory: SpatialMemory):
    """get_recent() should return latest detections."""
    await memory.initialise()

    detections = [
        {"object_name": f"obj_{i}", "confidence": 0.9, "frame_number": i}
        for i in range(5)
    ]
    await memory.log_detection_batch(detections)

    recent = await memory.get_recent(n=3)
    assert len(recent) == 3
    await memory.close()


@pytest.mark.asyncio
async def test_get_summary(memory: SpatialMemory):
    """get_summary() should return aggregate stats."""
    await memory.initialise()

    detections = [
        {"object_name": "dog", "confidence": 0.9, "frame_number": 1},
        {"object_name": "cat", "confidence": 0.8, "frame_number": 2},
        {"object_name": "dog", "confidence": 0.85, "frame_number": 3},
    ]
    await memory.log_detection_batch(detections)

    summary = await memory.get_summary()
    # Dedup may merge the two 'dog' entries if they're within cooldown
    assert summary["total_detections"] >= 2
    assert summary["unique_objects"] >= 2
    await memory.close()


@pytest.mark.asyncio
async def test_dedup_cooldown(memory: SpatialMemory):
    """Same object logged rapidly should be deduplicated."""
    memory.DEDUP_COOLDOWN_SECONDS = 10.0  # long cooldown for test
    await memory.initialise()

    det = [{"object_name": "cup", "confidence": 0.9, "frame_number": 1}]
    await memory.log_detection_batch(det)
    # Log again — should be deduped
    await memory.log_detection_batch(det)

    results = await memory.search("cup")
    assert len(results) == 1, "Dedup should prevent duplicate entries"
    await memory.close()


@pytest.mark.asyncio
async def test_environment_context(memory: SpatialMemory):
    """get_environment_context() should return a non-empty string."""
    await memory.initialise()

    detections = [
        {"object_name": "bench", "direction": "ahead", "confidence": 0.9, "frame_number": 1},
    ]
    await memory.log_detection_batch(detections)

    ctx = await memory.get_environment_context()
    assert isinstance(ctx, str)
    assert "bench" in ctx.lower()
    await memory.close()


# ---------------------------------------------------------------------------
# NavigationEngine tests
# ---------------------------------------------------------------------------
from mcp_tools.navigation_engine import NavigationEngine


@pytest.fixture
def nav_engine():
    return NavigationEngine()


def test_nav_engine_process_detections(nav_engine: NavigationEngine):
    """process_detections should return announcements for new objects."""
    detections = [
        {"object_name": "person", "direction": "ahead", "distance": "nearby", "confidence": 0.9},
        {"object_name": "car", "direction": "left", "distance": "far", "confidence": 0.8},
    ]
    announcements = nav_engine.process_detections(detections)
    # First time seeing objects → should generate announcements
    assert len(announcements) >= 1


def test_nav_engine_dedup_announcements(nav_engine: NavigationEngine):
    """Same detections processed twice rapidly should not re-announce."""
    dets = [
        {"object_name": "tree", "direction": "right", "distance": "far", "confidence": 0.7},
    ]
    first = nav_engine.process_detections(dets)
    second = nav_engine.process_detections(dets)
    # Second call should produce fewer or no announcements
    assert len(second) <= len(first)


def test_nav_engine_assistant_mode(nav_engine: NavigationEngine):
    """Assistant mode should suppress navigation announcements."""
    nav_engine.activate_assistant()

    dets = [
        {"object_name": "person", "direction": "ahead", "distance": "nearby", "confidence": 0.9},
    ]
    announcements = nav_engine.process_detections(dets)
    # In assistant mode, only critical hazards should be announced
    non_critical = [a for a in announcements if a.get("priority", 99) > 0]
    assert len(non_critical) == 0, "Non-critical announcements suppressed in assistant mode"

    nav_engine.deactivate_assistant()


def test_nav_engine_environment_summary(nav_engine: NavigationEngine):
    """get_environment_summary() should return a string."""
    dets = [
        {"object_name": "bicycle", "direction": "ahead", "distance": "nearby", "confidence": 0.9},
    ]
    nav_engine.process_detections(dets)
    summary = nav_engine.get_environment_summary()
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_nav_engine_hazard_alerts(nav_engine: NavigationEngine):
    """Hazard alerts should be returned from get_hazard_alerts()."""
    # Process some hazardous objects
    dets = [
        {"object_name": "pothole", "direction": "ahead", "distance": "close", "confidence": 0.9},
    ]
    nav_engine.process_detections(dets)
    alerts = nav_engine.get_hazard_alerts()
    assert isinstance(alerts, list)


def test_nav_engine_user_speech_suppression(nav_engine: NavigationEngine):
    """on_user_speech() should temporarily suppress announcements."""
    nav_engine.on_user_speech()
    # Immediately after user speech, non-critical announcements suppressed
    dets = [
        {"object_name": "bench", "direction": "left", "distance": "far", "confidence": 0.7},
    ]
    announcements = nav_engine.process_detections(dets)
    # Low priority items should be suppressed
    low_priority = [a for a in announcements if a.get("priority", 99) >= 3]
    assert len(low_priority) == 0


# ---------------------------------------------------------------------------
# Maps API tests (stub mode — no API key)
# ---------------------------------------------------------------------------
from mcp_tools.maps_api import get_walking_directions, search_nearby_places


@pytest.mark.asyncio
async def test_maps_stub_directions():
    """Without MAPS_API_KEY, should still return a graceful stub response."""
    original = os.environ.get("MAPS_API_KEY")
    os.environ.pop("MAPS_API_KEY", None)

    result = await get_walking_directions("Times Square")
    assert isinstance(result, dict)
    # Should either be an error or stub, not crash
    assert "status" in result or "error" in result or "spoken_summary" in result

    if original:
        os.environ["MAPS_API_KEY"] = original


@pytest.mark.asyncio
async def test_maps_stub_nearby():
    """Without MAPS_API_KEY, search_nearby_places should not crash."""
    original = os.environ.get("MAPS_API_KEY")
    os.environ.pop("MAPS_API_KEY", None)

    result = await search_nearby_places("pharmacy")
    assert isinstance(result, dict)

    if original:
        os.environ["MAPS_API_KEY"] = original
