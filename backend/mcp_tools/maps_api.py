"""
Google Maps Navigation Tool
=============================
Day 4: Full implementation with live Google Maps Directions API,
geocoding, nearby places search, and graceful fallback.

MCP tool for fetching walking directions. The agent calls this when a
user asks "How do I get to X?" or "Navigate to the nearest pharmacy."

Features:
  - Walking directions with turn-by-turn instructions
  - Geocoding (address → lat/lng) for precise locations
  - Nearby places search ("nearest pharmacy", "closest bus stop")
  - IP-based approximate location when GPS not available
  - Structured output with distance, duration, and clean instructions
  - Graceful fallback with known local routes when API key is missing
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Optional

import httpx

logger = logging.getLogger("mcp.maps")

# NOTE: Do NOT read the API key at module level — load_dotenv() in main.py
# runs AFTER this module is imported, so os.getenv() would return "".
# Use _get_api_key() at call time instead.
DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
PLACES_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
PLACES_TEXT_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
GEOLOCATE_URL = "https://www.googleapis.com/geolocation/v1/geolocate"

# Timeout for all Google Maps API calls
_TIMEOUT = httpx.Timeout(10.0, connect=5.0)

# ---------------------------------------------------------------------------
# Known campus location (L&T South City, Bangalore)
# Used as origin when IP geolocation is unreliable (e.g. VPN, hotspot)
# ---------------------------------------------------------------------------
_CAMPUS_LOCATION = {
    "lat": 12.886722,
    "lng": 77.5895493,
    "address": "L&T South City Block B7, Bangalore",
}

# ---------------------------------------------------------------------------
# Known short-distance routes (fallback when Maps can't resolve building IDs)
# ---------------------------------------------------------------------------
_KNOWN_ROUTES: dict[tuple[str, str], dict] = {
    ("b7", "b9"): {
        "status": "ok",
        "summary": "20 m — 1 min walk",
        "spoken_summary": (
            "Walking from B7 to B9. Total distance about 20 meters, "
            "roughly 1 minute walk. Head southeast along the pathway. "
            "Look for the B9 sign on the building ahead."
        ),
        "start_address": "L&T South City Block B7, Bangalore",
        "end_address": "L&T South City Block B9, Bangalore",
        "total_distance": "20 m",
        "total_duration": "1 min",
        "steps": [
            {"step_number": 1, "instruction": "Exit B7 and head southeast along the pathway", "distance": "15 m", "duration": "1 min"},
            {"step_number": 2, "instruction": "B9 will be on your right. Look for the B9 sign on the building", "distance": "5 m", "duration": "0 min"},
        ],
        "step_count": 2,
    },
    ("b9", "b7"): {
        "status": "ok",
        "summary": "20 m — 1 min walk",
        "spoken_summary": (
            "Walking from B9 to B7. Total distance about 20 meters, "
            "roughly 1 minute walk. Head northwest along the pathway. "
            "Look for the B7 sign on the building ahead."
        ),
        "start_address": "L&T South City Block B9, Bangalore",
        "end_address": "L&T South City Block B7, Bangalore",
        "total_distance": "20 m",
        "total_duration": "1 min",
        "steps": [
            {"step_number": 1, "instruction": "Exit B9 and head northwest along the pathway", "distance": "15 m", "duration": "1 min"},
            {"step_number": 2, "instruction": "B7 will be on your left. Look for the B7 sign on the building", "distance": "5 m", "duration": "0 min"},
        ],
        "step_count": 2,
    },
}


def _check_known_route(origin: str, destination: str) -> dict | None:
    """Check if origin→destination matches a known short route."""
    def _normalize(s: str) -> str:
        s = s.lower().strip()
        # Extract block identifiers like 'b7', 'b9'
        for token in s.replace(',', ' ').split():
            if re.match(r'^b\d+$', token):
                return token
        return s

    o = _normalize(origin)
    d = _normalize(destination)
    return _KNOWN_ROUTES.get((o, d))


# ---------------------------------------------------------------------------
# Quota / Rate-Limit Guards
# ---------------------------------------------------------------------------
# Google Maps Platform gives $200/month free credit.  Costs per 1,000 calls:
#   Directions       $5   → ~1,333/day  (40k/month)
#   Geocoding        $5   → ~1,333/day
#   Geolocation      $5   → ~1,333/day
#   Places Text      $32  → ~208/day    (6.25k/month)  ← most expensive
#
# We set conservative DAILY caps well below free-tier limits and a per-minute
# rate limit to prevent accidental bursts.

_DAILY_LIMITS: dict[str, int] = {
    "directions": 100,     # well under 1,333/day cap
    "geocode": 150,        # well under 1,333/day cap
    "geolocation": 100,    # well under 1,333/day cap
    "places": 50,          # well under 208/day cap (most expensive)
}

_PER_MINUTE_LIMIT = 10  # max total Maps API calls per minute (across all types)

# Mutable counters — reset daily
_daily_counts: dict[str, int] = {k: 0 for k in _DAILY_LIMITS}
_daily_reset_day: int = 0  # day-of-year of last reset

# Sliding-window minute tracker
_minute_timestamps: list[float] = []


def _reset_daily_if_needed() -> None:
    """Reset daily counters at the start of each new UTC day."""
    global _daily_reset_day
    today = int(time.time() // 86400)
    if today != _daily_reset_day:
        for k in _daily_counts:
            _daily_counts[k] = 0
        _daily_reset_day = today
        logger.info("Maps quota counters reset for new day")


def _check_rate_limit() -> bool:
    """Return True if we're within the per-minute rate limit."""
    now = time.monotonic()
    # Purge timestamps older than 60s
    while _minute_timestamps and _minute_timestamps[0] < now - 60:
        _minute_timestamps.pop(0)
    return len(_minute_timestamps) < _PER_MINUTE_LIMIT


def _record_call(api_type: str) -> None:
    """Record a Maps API call for quota tracking."""
    _daily_counts[api_type] = _daily_counts.get(api_type, 0) + 1
    _minute_timestamps.append(time.monotonic())


def _can_call(api_type: str) -> tuple[bool, str]:
    """
    Check whether we're allowed to make a Maps API call of the given type.
    Returns (allowed, reason).
    """
    _reset_daily_if_needed()

    limit = _DAILY_LIMITS.get(api_type, 100)
    count = _daily_counts.get(api_type, 0)
    if count >= limit:
        msg = (
            f"Daily quota for {api_type} reached ({count}/{limit}). "
            "Try again tomorrow to stay within free-tier limits."
        )
        logger.warning(msg)
        return False, msg

    if not _check_rate_limit():
        msg = (
            f"Rate limit: more than {_PER_MINUTE_LIMIT} Maps API calls in "
            "the last minute. Please wait a moment."
        )
        logger.warning(msg)
        return False, msg

    return True, ""


def get_maps_quota_status() -> dict:
    """Return current quota usage (exposed to telemetry)."""
    _reset_daily_if_needed()
    return {
        api: {"used": _daily_counts.get(api, 0), "limit": lim}
        for api, lim in _DAILY_LIMITS.items()
    }


def _get_api_key() -> str:
    """Read the Maps API key at call time (after load_dotenv)."""
    return os.getenv("MAPS_API_KEY", "")


def _has_valid_key() -> bool:
    """Check if a valid Maps API key is configured."""
    key = _get_api_key()
    return bool(key) and not key.startswith("your_")


def _clean_html(html_text: str) -> str:
    """Strip HTML tags from Google Maps instructions."""
    return re.sub(r"<[^>]+>", " ", html_text).strip()


def _clean_html_for_speech(html_text: str) -> str:
    """
    Convert HTML directions to clean speech-friendly text.
    Converts tags like <b> to emphasis and strips the rest.
    """
    # Replace <b>...</b> with the content (no tags)
    text = re.sub(r"<b>(.*?)</b>", r"\1", html_text)
    # Replace <div> with a period-space to create natural sentence breaks
    text = re.sub(r"<div[^>]*>", ". ", text)
    # Strip all remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Fix double periods
    text = text.replace("..", ".").replace(". .", ".")
    return text


# ---------------------------------------------------------------------------
# Core API functions
# ---------------------------------------------------------------------------

async def get_walking_directions(
    destination: str,
    origin: str = "current location",
) -> dict:
    """
    Fetch step-by-step walking directions from Google Maps.

    Args:
        destination: Where the user wants to go (address, place, or
            "nearest X" for nearby places search).
        origin: Starting point (default: "current location" — the agent
            should describe what the user means, or use IP geolocation).

    Returns:
        Dict with route summary, step-by-step directions, and metadata.
    """
    if not _has_valid_key():
        logger.warning("MAPS_API_KEY not set or invalid — using local route data")
        return _stub_directions(destination)

    logger.info("Maps API key found — will use live Google Maps Directions API")

    # Check known local routes first (e.g. B7→B9 on campus)
    known = _check_known_route(origin, destination)
    if known:
        logger.info("Using known route for '%s' → '%s'", origin, destination)
        return known

    # Quota guard — directions call (+ possibly geocode & places)
    ok, reason = _can_call("directions")
    if not ok:
        return {"status": "quota_exceeded", "message": reason}

    try:
        # Handle "nearest X" / "closest X" queries via Places API
        if _is_nearby_query(destination):
            place_type = _extract_place_type(destination)
            place_result = await search_nearby_places(place_type, origin)
            if place_result.get("status") == "ok" and place_result.get("places"):
                # Use first result as the actual destination
                first_place = place_result["places"][0]
                destination = first_place.get("address", first_place.get("name", destination))
                logger.info(
                    "Resolved 'nearest %s' → %s", place_type, destination
                )

        # Resolve origin — use campus coordinates as default (more reliable
        # than IP geolocation which often resolves to wrong city on hotspot)
        actual_origin = origin
        if origin.lower() in ("current location", "here", "my location"):
            actual_origin = f"{_CAMPUS_LOCATION['lat']},{_CAMPUS_LOCATION['lng']}"
            logger.info("Using campus location as origin: %s", actual_origin)

        _record_call("directions")
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                DIRECTIONS_URL,
                params={
                    "origin": actual_origin,
                    "destination": destination,
                    "mode": "walking",
                    "key": _get_api_key(),
                    "language": "en",
                },
            )
            data = resp.json()

        if data.get("status") != "OK":
            error_msg = data.get("status", "Unknown error")
            # Provide helpful error messages
            if error_msg == "ZERO_RESULTS":
                return {
                    "status": "no_route",
                    "message": f"Sorry, I couldn't find walking directions to '{destination}'. "
                               "It might be too far to walk, or the location wasn't found.",
                }
            elif error_msg == "NOT_FOUND":
                return {
                    "status": "not_found",
                    "message": f"I couldn't find the location '{destination}'. "
                               "Could you be more specific?",
                }
            return {"status": "error", "message": error_msg}

        route = data["routes"][0]
        leg = route["legs"][0]

        # Build speech-friendly step-by-step directions
        steps = []
        for i, step in enumerate(leg["steps"], 1):
            instruction = step.get("html_instructions", "")
            clean = _clean_html_for_speech(instruction)
            steps.append({
                "step_number": i,
                "instruction": clean,
                "distance": step["distance"]["text"],
                "duration": step["duration"]["text"],
            })

        # Build a concise spoken summary
        total_distance = leg["distance"]["text"]
        total_duration = leg["duration"]["text"]
        first_step = steps[0]["instruction"] if steps else "Start walking"

        spoken_summary = (
            f"Walking to {leg['end_address']}. "
            f"Total distance: {total_distance}, about {total_duration}. "
            f"To start: {first_step}."
        )

        return {
            "status": "ok",
            "summary": f"{total_distance} — {total_duration}",
            "spoken_summary": spoken_summary,
            "start_address": leg.get("start_address", actual_origin),
            "end_address": leg.get("end_address", destination),
            "total_distance": total_distance,
            "total_duration": total_duration,
            "steps": steps,
            "step_count": len(steps),
        }

    except httpx.TimeoutException:
        logger.error("Maps API timed out for destination: %s", destination)
        return {
            "status": "timeout",
            "message": "The directions service is taking too long. Please try again.",
        }
    except Exception as e:
        logger.error("Maps API error: %s", e)
        return {"status": "error", "message": str(e)}


async def search_nearby_places(
    place_type: str,
    location: str = "current location",
    radius: int = 1000,
) -> dict:
    """
    Search for nearby places of a given type.

    Args:
        place_type: Type of place (e.g., "pharmacy", "bus stop", "restaurant").
        location: Center point for the search.
        radius: Search radius in meters.

    Returns:
        Dict with list of nearby places and their details.
    """
    if not _has_valid_key():
        return {
            "status": "error",
            "message": f"Could not search for nearby {place_type} at this time. Please try again later.",
            "places": [],
        }

    # Quota guard — places is the most expensive API ($32/1k)
    ok, reason = _can_call("places")
    if not ok:
        return {"status": "quota_exceeded", "message": reason}

    try:
        # Resolve location
        lat, lng = None, None
        if location.lower() in ("current location", "here", "my location"):
            # Use campus coordinates (IP geolocation unreliable on hotspot)
            lat = _CAMPUS_LOCATION["lat"]
            lng = _CAMPUS_LOCATION["lng"]
        else:
            coords = await _geocode(location)
            if coords:
                lat, lng = coords["lat"], coords["lng"]

        if lat is None or lng is None:
            return {
                "status": "error",
                "message": "Could not determine location for nearby search.",
            }

        _record_call("places")
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                PLACES_TEXT_URL,
                params={
                    "query": place_type,
                    "location": f"{lat},{lng}",
                    "radius": radius,
                    "key": _get_api_key(),
                },
            )
            data = resp.json()

        if data.get("status") not in ("OK", "ZERO_RESULTS"):
            return {"status": "error", "message": data.get("status", "Unknown error")}

        places = []
        for result in data.get("results", [])[:5]:
            place = {
                "name": result.get("name", ""),
                "address": result.get("formatted_address", ""),
                "rating": result.get("rating"),
                "open_now": result.get("opening_hours", {}).get("open_now"),
                "place_id": result.get("place_id", ""),
            }
            # Calculate approximate distance
            place_lat = result["geometry"]["location"]["lat"]
            place_lng = result["geometry"]["location"]["lng"]
            dist_m = _haversine_distance(lat, lng, place_lat, place_lng)
            place["distance_meters"] = int(dist_m)
            place["distance"] = (
                f"{int(dist_m)}m" if dist_m < 1000 else f"{dist_m / 1000:.1f}km"
            )
            places.append(place)

        # Sort by distance
        places.sort(key=lambda p: p.get("distance_meters", 99999))

        return {
            "status": "ok",
            "query": place_type,
            "places": places,
            "count": len(places),
        }

    except Exception as e:
        logger.error("Nearby places search error: %s", e)
        return {"status": "error", "message": str(e)}


async def get_current_location_info() -> dict:
    """
    Get approximate current location info via IP geolocation.
    Used when GPS is not available.
    """
    geo = await _geolocate_ip()
    if not geo:
        return {"status": "error", "message": "Could not determine location"}

    # Reverse geocode to get address
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            if _has_valid_key():
                ok, _ = _can_call("geocode")
                if not ok:
                    # Skip reverse geocode, return coords only
                    return {
                        "status": "ok",
                        "lat": geo["lat"],
                        "lng": geo["lng"],
                        "address": "Approximate location (quota limit reached)",
                        "accuracy": geo.get("accuracy", "unknown"),
                    }
                _record_call("geocode")
                resp = await client.get(
                    GEOCODE_URL,
                    params={
                        "latlng": f"{geo['lat']},{geo['lng']}",
                        "key": _get_api_key(),
                    },
                )
                data = resp.json()
                if data.get("status") == "OK" and data.get("results"):
                    address = data["results"][0].get("formatted_address", "")
                    return {
                        "status": "ok",
                        "lat": geo["lat"],
                        "lng": geo["lng"],
                        "address": address,
                        "accuracy": geo.get("accuracy", "unknown"),
                    }
    except Exception:
        pass

    return {
        "status": "ok",
        "lat": geo["lat"],
        "lng": geo["lng"],
        "address": "Approximate location (IP-based)",
        "accuracy": geo.get("accuracy", "unknown"),
    }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _is_nearby_query(destination: str) -> bool:
    """Check if destination is a 'nearest/closest X' type query."""
    d = destination.lower().strip()
    return any(d.startswith(prefix) for prefix in (
        "nearest", "closest", "nearby", "find a", "find the nearest",
        "where is the nearest", "where is the closest",
    ))


def _extract_place_type(query: str) -> str:
    """Extract the place type from a 'nearest X' query."""
    q = query.lower().strip()
    for prefix in (
        "where is the nearest", "where is the closest",
        "find the nearest", "find the closest",
        "nearest", "closest", "nearby", "find a",
    ):
        if q.startswith(prefix):
            return q[len(prefix):].strip()
    return q


async def _geocode(address: str) -> Optional[dict]:
    """Geocode an address to lat/lng coordinates."""
    if not _has_valid_key():
        return None
    ok, _ = _can_call("geocode")
    if not ok:
        return None
    try:
        _record_call("geocode")
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                GEOCODE_URL,
                params={"address": address, "key": _get_api_key()},
            )
            data = resp.json()
            if data.get("status") == "OK" and data.get("results"):
                loc = data["results"][0]["geometry"]["location"]
                return {"lat": loc["lat"], "lng": loc["lng"]}
    except Exception as e:
        logger.error("Geocode error: %s", e)
    return None


async def _geolocate_ip() -> Optional[dict]:
    """Get approximate location via IP geolocation (free fallback)."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Try Google geolocation API first (if key available)
            if _has_valid_key():
                ok, _ = _can_call("geolocation")
                if ok:
                    _record_call("geolocation")
                    resp = await client.post(
                        f"{GEOLOCATE_URL}?key={_get_api_key()}",
                        json={"considerIp": True},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        loc = data.get("location", {})
                        return {
                            "lat": loc.get("lat"),
                            "lng": loc.get("lng"),
                            "accuracy": data.get("accuracy", 0),
                        }

            # Fallback: ipinfo.io (free, no key needed)
            resp = await client.get("https://ipinfo.io/json")
            if resp.status_code == 200:
                data = resp.json()
                loc_str = data.get("loc", "")
                if "," in loc_str:
                    lat, lng = loc_str.split(",")
                    return {
                        "lat": float(lat),
                        "lng": float(lng),
                        "accuracy": 5000,
                        "city": data.get("city", ""),
                    }
    except Exception as e:
        logger.debug("IP geolocation failed: %s", e)
    return None


def _haversine_distance(
    lat1: float, lng1: float, lat2: float, lng2: float
) -> float:
    """Calculate distance between two points in meters (Haversine formula)."""
    import math

    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lng2 - lng1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _stub_directions(destination: str) -> dict:
    """Return fallback directions when API key is not configured."""
    # Check known routes first
    known = _check_known_route("current location", destination)
    if known:
        return known

    return {
        "status": "ok",
        "spoken_summary": (
            f"I have basic directions to {destination}. "
            "Head straight and look for signs to confirm you're on track. "
            "I'll keep watching for obstacles and read any signs I see."
        ),
        "steps": [
            {"step_number": 1, "instruction": f"Head towards {destination}", "distance": "unknown", "duration": "unknown"},
            {"step_number": 2, "instruction": "Look for signs to confirm your destination", "distance": "", "duration": ""},
        ],
        "step_count": 2,
        "total_distance": "unknown",
        "total_duration": "a few minutes",
    }
