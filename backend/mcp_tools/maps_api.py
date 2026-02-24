"""
Google Maps Navigation Tool
=============================
MCP tool for fetching walking directions via Google Maps Directions API.
The agent calls this when a user asks "How do I get to X?"

Day 4: Full implementation.
Day 1: Interface scaffold.
"""

import logging
import os

import httpx

logger = logging.getLogger("mcp.maps")

MAPS_API_KEY = os.getenv("MAPS_API_KEY", "")
DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"


async def get_walking_directions(
    destination: str,
    origin: str = "current location",
) -> dict:
    """
    Fetch step-by-step walking directions from Google Maps.

    Args:
        destination: Where the user wants to go.
        origin: Starting point (default: "current location" → uses GPS/IP).

    Returns:
        Dict with route summary and turn-by-turn directions.
    """
    if not MAPS_API_KEY or MAPS_API_KEY.startswith("your_"):
        logger.warning("MAPS_API_KEY not configured — returning stub")
        return {
            "status": "stub",
            "message": f"Directions to '{destination}' require a valid MAPS_API_KEY.",
            "steps": [
                "Walk north for 100 meters",
                "Turn right at the intersection",
                "Your destination will be on the left",
            ],
        }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                DIRECTIONS_URL,
                params={
                    "origin": origin,
                    "destination": destination,
                    "mode": "walking",
                    "key": MAPS_API_KEY,
                },
            )
            data = resp.json()

        if data.get("status") != "OK":
            return {"status": "error", "message": data.get("status", "Unknown error")}

        route = data["routes"][0]
        leg = route["legs"][0]
        steps = []
        for step in leg["steps"]:
            # Strip HTML tags from instructions
            instruction = step.get("html_instructions", "")
            import re
            clean = re.sub(r"<[^>]+>", " ", instruction).strip()
            steps.append({
                "instruction": clean,
                "distance": step["distance"]["text"],
                "duration": step["duration"]["text"],
            })

        return {
            "status": "ok",
            "summary": f"{leg['distance']['text']} — {leg['duration']['text']}",
            "steps": steps,
        }

    except Exception as e:
        logger.error("Maps API error: %s", e)
        return {"status": "error", "message": str(e)}
