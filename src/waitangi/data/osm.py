"""OpenStreetMap data fetching for real river geometry."""

import json
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np


@dataclass
class WaitangiLandmarks:
    """Key landmarks for the Waitangi River area."""

    # Waitangi Bridge - spans tidal channel between Paihia and Waitangi
    # Bridge connects land on both sides over the tidal channel
    BRIDGE_PAIHIA_LAT = -35.272709
    BRIDGE_PAIHIA_LON = 174.079625
    BRIDGE_WAITANGI_LAT = -35.271170
    BRIDGE_WAITANGI_LON = 174.079652
    # Center point of bridge
    BRIDGE_LAT = -35.271940
    BRIDGE_LON = 174.079638

    # Boat ramp - actual launch point
    SLIPWAY_LAT = -35.270798
    SLIPWAY_LON = 174.078968

    # Tidal channel mouth (where channel meets the bay, near bridge)
    MOUTH_LAT = -35.2708
    MOUTH_LON = 174.0790

    # Haruru Falls - maximum kayaking extent (small waterfall)
    HARURU_FALLS_LAT = -35.278284
    HARURU_FALLS_LON = 174.051297

    # Upstream extent (where river data ends, beyond falls)
    UPSTREAM_LAT = -35.3164
    UPSTREAM_LON = 173.8221


OVERPASS_API = "https://overpass-api.de/api/interpreter"

WAITANGI_QUERY = """
[out:json];
(
  way["waterway"="river"]["name"~"Waitangi",i](-35.35,173.80,-35.24,174.15);
  way["natural"="water"](-35.30,174.04,-35.24,174.13);
  way["waterway"~"river|stream|tidal_channel"](-35.30,174.04,-35.24,174.13);
  way["natural"="coastline"](-35.30,174.04,-35.24,174.15);
  way["bridge"="yes"](-35.28,174.07,-35.26,174.10);
  node["leisure"="slipway"](-35.28,174.07,-35.26,174.10);
  relation["natural"="water"](-35.30,174.04,-35.24,174.13);
);
out geom;
"""


async def fetch_waitangi_geometry(cache_path: Path | None = None) -> dict:
    """Fetch Waitangi River geometry from OpenStreetMap.

    Args:
        cache_path: Optional path to cache the data.

    Returns:
        Dictionary with river, coastline, and landmark data.
    """
    # Check cache first
    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            OVERPASS_API,
            data={"data": WAITANGI_QUERY},
        )
        response.raise_for_status()
        data = response.json()

    result = parse_osm_data(data)

    # Cache if path provided
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(result, f)

    return result


def fetch_waitangi_geometry_sync(cache_path: Path | None = None) -> dict:
    """Synchronous version of fetch_waitangi_geometry."""
    import asyncio
    return asyncio.run(fetch_waitangi_geometry(cache_path))


def parse_osm_data(data: dict) -> dict:
    """Parse OSM Overpass API response into usable geometry.

    Returns:
        Dictionary with:
        - river_segments: List of coordinate arrays for river/waterway centerlines
        - estuary_polygons: List of coordinate arrays for estuary/tidal water bodies
        - coastline_segments: List of coordinate arrays for coastline
        - bridge: Bridge location and geometry
        - slipway: Boat ramp location
    """
    river_segments = []
    estuary_polygons = []
    coastline_segments = []
    water_polygons = []
    bridge = None
    slipway = None

    for el in data.get("elements", []):
        tags = el.get("tags", {})
        geom = el.get("geometry", [])

        if el["type"] == "node" and tags.get("leisure") == "slipway":
            slipway = {
                "lat": el["lat"],
                "lon": el["lon"],
            }

        elif el["type"] == "way":
            coords = [(pt["lat"], pt["lon"]) for pt in geom]

            # River/stream/waterway - linear features
            waterway = tags.get("waterway", "")
            if waterway in ("river", "stream", "tidal_channel"):
                river_segments.append(coords)

            # Coastline
            elif tags.get("natural") == "coastline":
                coastline_segments.append(coords)

            # Water bodies (polygons) - estuaries, bays, tidal areas
            elif tags.get("natural") == "water" or tags.get("natural") == "bay":
                # Check if it's a closed polygon (estuary/lake) vs open water
                water_type = tags.get("water", "")
                if water_type in ("river", "tidal", "lagoon", "estuary", "bay", "cove"):
                    estuary_polygons.append(coords)
                else:
                    water_polygons.append(coords)

            # Any bridge in the area
            elif tags.get("bridge") == "yes":
                bridge = {
                    "name": tags.get("name", "Bridge"),
                    "coords": coords,
                }

        # Also handle relations (for complex water bodies)
        elif el["type"] == "relation":
            if tags.get("natural") == "water":
                # Extract outer ways from relation
                for member in el.get("members", []):
                    if member.get("role") == "outer" and member.get("geometry"):
                        coords = [(pt["lat"], pt["lon"]) for pt in member["geometry"]]
                        estuary_polygons.append(coords)

    # Merge river segments into continuous centerline
    river_centerline = merge_river_segments(river_segments)

    return {
        "river_centerline": river_centerline,
        "river_segments": river_segments,
        "estuary_polygons": estuary_polygons,
        "coastline_segments": coastline_segments,
        "water_polygons": water_polygons,
        "bridge": bridge,
        "slipway": slipway,
        "landmarks": {
            "bridge": (WaitangiLandmarks.BRIDGE_LAT, WaitangiLandmarks.BRIDGE_LON),
            "bridge_paihia": (WaitangiLandmarks.BRIDGE_PAIHIA_LAT, WaitangiLandmarks.BRIDGE_PAIHIA_LON),
            "bridge_waitangi": (WaitangiLandmarks.BRIDGE_WAITANGI_LAT, WaitangiLandmarks.BRIDGE_WAITANGI_LON),
            "slipway": (WaitangiLandmarks.SLIPWAY_LAT, WaitangiLandmarks.SLIPWAY_LON),
            "mouth": (WaitangiLandmarks.MOUTH_LAT, WaitangiLandmarks.MOUTH_LON),
            "haruru_falls": (WaitangiLandmarks.HARURU_FALLS_LAT, WaitangiLandmarks.HARURU_FALLS_LON),
        },
    }


def merge_river_segments(segments: list[list[tuple]]) -> list[tuple]:
    """Merge river segments into a single continuous centerline.

    Segments are ordered from mouth to upstream.
    """
    if not segments:
        return []

    # Sort segments by starting longitude (mouth is easternmost)
    segments = sorted(segments, key=lambda s: -s[0][1] if s else 0)

    merged = []
    for seg in segments:
        if not seg:
            continue

        if not merged:
            merged.extend(seg)
        else:
            # Check if this segment connects to the end of merged
            last_point = merged[-1]
            first_point = seg[0]
            last_point_rev = seg[-1]

            dist_forward = _haversine_distance(last_point, first_point)
            dist_reverse = _haversine_distance(last_point, last_point_rev)

            if dist_reverse < dist_forward:
                # Segment is reversed
                seg = list(reversed(seg))

            # Skip first point if it's close to last (avoid duplicates)
            if _haversine_distance(merged[-1], seg[0]) < 50:  # 50m threshold
                merged.extend(seg[1:])
            else:
                merged.extend(seg)

    return merged


def _haversine_distance(p1: tuple, p2: tuple) -> float:
    """Calculate distance between two lat/lon points in meters."""
    lat1, lon1 = np.radians(p1)
    lat2, lon2 = np.radians(p2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Earth radius in meters
    return 6371000 * c


def get_cached_geometry() -> dict | None:
    """Load cached geometry if available."""
    from waitangi.core.config import get_settings

    settings = get_settings()
    cache_path = settings.data_sources.cache_dir / "waitangi_geometry.json"

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return None


def save_geometry_cache(data: dict) -> None:
    """Save geometry to cache."""
    from waitangi.core.config import get_settings

    settings = get_settings()
    cache_path = settings.data_sources.cache_dir / "waitangi_geometry.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_path, "w") as f:
        json.dump(data, f)
