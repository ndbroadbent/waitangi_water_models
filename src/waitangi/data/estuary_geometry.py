"""Waitangi estuary geometry from LINZ coastline data.

Provides water/land boundaries using authoritative LINZ Mean High Water
coastline data, validated against ground-truth reference points.
"""

from functools import lru_cache
from typing import TYPE_CHECKING

from pyproj import Transformer
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    shape,
)
from shapely.ops import linemerge, polygonize, unary_union

from waitangi.core.config import WaitangiLocation
from waitangi.data.linz import LINZClient, WAITANGI_BBOX_WGS84
from waitangi.data.reference_points import (
    LANDMARKS,
    LAND_POINTS,
    WATER_POINTS_EAST,
    WATER_POINTS_WEST,
)

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry


def _get_transformer_to_nztm() -> Transformer:
    """Get WGS84 -> NZTM transformer."""
    return Transformer.from_crs(
        WaitangiLocation.CRS_WGS84, WaitangiLocation.CRS_NZTM, always_xy=True
    )


def _get_transformer_to_wgs84() -> Transformer:
    """Get NZTM -> WGS84 transformer."""
    return Transformer.from_crs(
        WaitangiLocation.CRS_NZTM, WaitangiLocation.CRS_WGS84, always_xy=True
    )


@lru_cache(maxsize=1)
def get_linz_coastlines_nztm() -> list[LineString | MultiLineString]:
    """Fetch and cache LINZ Mean High Water coastlines.

    Returns list of LineString/MultiLineString geometries in NZTM coordinates.
    """
    client = LINZClient()
    data = client.fetch_coastline_mhw()
    features = data.get("features", [])

    geometries = []
    for feature in features:
        geom = shape(feature["geometry"])
        geometries.append(geom)

    return geometries


@lru_cache(maxsize=1)
def get_coastline_merged_nztm() -> MultiLineString | LineString:
    """Get all coastlines merged into a single geometry."""
    coastlines = get_linz_coastlines_nztm()
    if not coastlines:
        raise ValueError("No coastline data available from LINZ")

    # Merge all coastline segments
    merged = linemerge(coastlines)
    return merged


def get_coastline_bbox_nztm() -> tuple[float, float, float, float]:
    """Get bounding box of coastline data in NZTM."""
    coastlines = get_linz_coastlines_nztm()
    if not coastlines:
        raise ValueError("No coastline data")

    all_geoms = unary_union(coastlines)
    return all_geoms.bounds  # (minx, miny, maxx, maxy)


# User-specified bounding boxes (WGS84: min_lon, min_lat, max_lon, max_lat)
# Full visualization bounds (estuary + tributaries + mangroves)
# Top-left: -35.262348, 174.049555 / Bottom-right: -35.285649, 174.085256
VISUALIZATION_BBOX_WGS84 = (174.049555, -35.285649, 174.085256, -35.262348)

# Focused kayak route bounds (typical paddle from boat ramp to Haruru Falls)
# Top-left: -35.268067, 174.055135 / Bottom-right: -35.277083, 174.082409
KAYAK_ROUTE_BBOX_WGS84 = (174.055135, -35.277083, 174.082409, -35.268067)


def get_water_polygon_nztm() -> Polygon | MultiPolygon:
    """Build water polygon using flood-fill approach from known water points.

    Strategy:
    1. Create a bounding box for the simulation area
    2. Use coastlines to create barriers (land boundaries)
    3. Flood fill from known water point, stopping at coastlines
    4. Result is the navigable water area

    Returns:
        Polygon or MultiPolygon representing navigable water area.
    """
    return _flood_fill_water_area()


@lru_cache(maxsize=1)
def _flood_fill_water_area() -> Polygon | MultiPolygon:
    """Build water polygon using flood-fill approach.

    Algorithm:
    1. Create bounding box from user-specified visualization bounds
    2. Use LINZ coastlines as barriers
    3. Split the bounding box using coastlines (polygonize)
    4. Keep polygons that contain known water reference points
    """
    transformer = _get_transformer_to_nztm()

    # Convert visualization bbox to NZTM
    min_lon, min_lat, max_lon, max_lat = VISUALIZATION_BBOX_WGS84
    min_e, min_n = transformer.transform(min_lon, min_lat)
    max_e, max_n = transformer.transform(max_lon, max_lat)

    # Create bounding box polygon
    bbox = Polygon([
        (min_e, min_n),
        (max_e, min_n),
        (max_e, max_n),
        (min_e, max_n),
        (min_e, min_n),
    ])

    # Get coastlines and merge them
    coastlines = get_linz_coastlines_nztm()
    if not coastlines:
        raise ValueError("No coastline data available")

    all_lines = unary_union(coastlines)

    # Add bbox boundary to the lines so we get closed regions
    bbox_boundary = bbox.boundary
    combined_lines = unary_union([all_lines, bbox_boundary])

    # Polygonize: split the plane into regions bounded by lines
    polygons = list(polygonize(combined_lines))

    if not polygons:
        # Fallback if polygonization fails
        return _build_water_from_reference_points()

    # Get water reference points in NZTM
    water_pts_nztm = [
        Point(transformer.transform(pt.lon, pt.lat))
        for pt in WATER_POINTS_WEST + WATER_POINTS_EAST
    ]

    # Keep only polygons that:
    # 1. Are inside or intersect the bounding box
    # 2. Contain at least one water reference point
    water_polygons = []
    for poly in polygons:
        if not bbox.intersects(poly):
            continue

        # Clip to bbox
        clipped = poly.intersection(bbox)
        if clipped.is_empty or clipped.area < 100:  # Skip tiny slivers
            continue

        # Check if this polygon contains any water reference point
        contains_water = any(
            clipped.contains(pt) or clipped.distance(pt) < 5
            for pt in water_pts_nztm
        )

        if contains_water:
            water_polygons.append(clipped)

    if not water_polygons:
        # No polygons contain water points - fallback
        return _build_water_from_reference_points()

    # Merge all water polygons
    result = unary_union(water_polygons)

    # Ensure we return Polygon or MultiPolygon
    if result.geom_type == "GeometryCollection":
        # Extract only polygons from collection
        polys = [g for g in result.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if polys:
            result = unary_union(polys)
        else:
            return _build_water_from_reference_points()

    return result


def _build_water_from_reference_points() -> Polygon | MultiPolygon:
    """Build water polygon from reference points when coastline polygonization fails.

    Uses known water points and coastline data to construct
    a valid water boundary.
    """
    transformer = _get_transformer_to_nztm()
    coastlines = get_linz_coastlines_nztm()
    all_lines = unary_union(coastlines) if coastlines else None

    # Convert water points to NZTM
    water_pts_nztm = [
        Point(transformer.transform(pt.lon, pt.lat))
        for pt in WATER_POINTS_WEST + WATER_POINTS_EAST
    ]

    # Create convex hull of water points as base water area
    from shapely.geometry import MultiPoint
    water_hull = MultiPoint(water_pts_nztm).convex_hull

    # Expand hull slightly to ensure all water points are included
    water_area = water_hull.buffer(50)

    # If we have coastlines, use them to refine the boundary
    if all_lines:
        # Intersect with a larger buffer of coastlines to stay within bounds
        coastline_buffer = all_lines.buffer(200)
        water_area = water_area.intersection(coastline_buffer)

    return water_area


def get_estuary_water_area_nztm() -> Polygon | MultiPolygon:
    """Get the estuary water area focused on the Waitangi tidal zone.

    Uses polygonization of LINZ coastlines to identify water bodies,
    falling back to reference point-based construction if needed.
    """
    return get_water_polygon_nztm()


@lru_cache(maxsize=1)
def get_estuary_polygon_nztm() -> Polygon | MultiPolygon:
    """Get estuary water area as Shapely geometry in NZTM coordinates.

    Primary method for getting the navigable water polygon.
    Uses LINZ coastline data with reference point validation.
    """
    return get_estuary_water_area_nztm()


def get_navigable_centerline_wgs84() -> list[tuple[float, float]]:
    """Get the main navigable channel centerline from bridge to Haruru Falls.

    Returns list of (lat, lon) tuples.
    """
    centerline = [
        (-35.2708, 174.0790),  # Mouth/bridge area (boat ramp)
        (-35.2710, 174.0770),  # Just west of bridge
        (-35.2700, 174.0720),
        (-35.2700, 174.0680),
        (-35.2720, 174.0650),
        (-35.2740, 174.0600),
        (-35.2750, 174.0550),
        (-35.2770, 174.0520),
        (-35.2783, 174.0513),  # Near Haruru Falls
    ]
    return centerline


def get_navigable_centerline_nztm() -> LineString:
    """Get centerline as Shapely LineString in NZTM."""
    transformer = _get_transformer_to_nztm()
    centerline_wgs84 = get_navigable_centerline_wgs84()
    centerline_nztm = [
        transformer.transform(lon, lat) for lat, lon in centerline_wgs84
    ]
    return LineString(centerline_nztm)


def point_to_nztm(lat: float, lon: float) -> tuple[float, float]:
    """Convert WGS84 lat/lon to NZTM coordinates."""
    transformer = _get_transformer_to_nztm()
    return transformer.transform(lon, lat)


def point_to_wgs84(easting: float, northing: float) -> tuple[float, float]:
    """Convert NZTM to WGS84 (returns lon, lat)."""
    transformer = _get_transformer_to_wgs84()
    return transformer.transform(easting, northing)


def is_in_water(lat: float, lon: float) -> bool:
    """Check if a WGS84 point is in the water area."""
    estuary = get_estuary_polygon_nztm()
    pt_nztm = Point(point_to_nztm(lat, lon))
    return estuary.contains(pt_nztm) or estuary.touches(pt_nztm)


def distance_to_shore(lat: float, lon: float) -> float:
    """Get distance from point to nearest shoreline in meters.

    Positive = in water, negative = on land.
    """
    estuary = get_estuary_polygon_nztm()
    pt_nztm = Point(point_to_nztm(lat, lon))

    if estuary.contains(pt_nztm):
        return estuary.boundary.distance(pt_nztm)
    else:
        return -estuary.boundary.distance(pt_nztm)


def validate_geometry() -> dict:
    """Validate the estuary geometry against reference points.

    Returns validation results showing which reference points
    are correctly classified as water or land.
    """
    transformer = _get_transformer_to_nztm()
    estuary = get_estuary_polygon_nztm()

    results = {
        "water_correct": 0,
        "water_incorrect": 0,
        "land_correct": 0,
        "land_incorrect": 0,
        "details": [],
    }

    # Check water points (should be inside estuary)
    for pt in WATER_POINTS_WEST + WATER_POINTS_EAST:
        pt_nztm = Point(transformer.transform(pt.lon, pt.lat))
        in_estuary = estuary.contains(pt_nztm) or estuary.distance(pt_nztm) < 10

        if in_estuary:
            results["water_correct"] += 1
            results["details"].append((pt.name, "water", "OK"))
        else:
            dist = estuary.boundary.distance(pt_nztm)
            results["water_incorrect"] += 1
            results["details"].append((pt.name, "water", f"FAIL ({dist:.0f}m away)"))

    # Check land points (should be outside estuary)
    for pt in LAND_POINTS:
        pt_nztm = Point(transformer.transform(pt.lon, pt.lat))
        in_estuary = estuary.contains(pt_nztm)

        if not in_estuary:
            results["land_correct"] += 1
            results["details"].append((pt.name, "land", "OK"))
        else:
            results["land_incorrect"] += 1
            results["details"].append((pt.name, "land", "FAIL (in water!)"))

    return results


def get_simulation_bounds_nztm() -> tuple[float, float, float, float]:
    """Get bounds for the simulation area in NZTM.

    Returns (min_easting, min_northing, max_easting, max_northing).
    """
    transformer = _get_transformer_to_nztm()

    # Convert WGS84 bbox to NZTM
    min_lon, min_lat, max_lon, max_lat = WAITANGI_BBOX_WGS84
    min_e, min_n = transformer.transform(min_lon, min_lat)
    max_e, max_n = transformer.transform(max_lon, max_lat)

    return (min_e, min_n, max_e, max_n)


def get_simulation_bounds_wgs84() -> tuple[float, float, float, float]:
    """Get bounds for the simulation area in WGS84.

    Returns (min_lon, min_lat, max_lon, max_lat).
    """
    return WAITANGI_BBOX_WGS84


if __name__ == "__main__":
    print("=== LINZ COASTLINE DATA ===\n")

    try:
        coastlines = get_linz_coastlines_nztm()
        print(f"Loaded {len(coastlines)} coastline features")

        bounds = get_coastline_bbox_nztm()
        print(f"Bounds (NZTM): {bounds}")

        estuary = get_estuary_polygon_nztm()
        print(f"Estuary geometry type: {estuary.geom_type}")
        print(f"Estuary area: {estuary.area / 1e6:.2f} km²")

    except ValueError as e:
        print(f"Error loading LINZ data: {e}")
        print("Make sure LINZ API key is set in .env")

    print("\n=== GEOMETRY VALIDATION ===\n")

    try:
        results = validate_geometry()

        for name, ptype, status in results["details"]:
            print(f"  {name} ({ptype}): {status}")

        total_water = results["water_correct"] + results["water_incorrect"]
        total_land = results["land_correct"] + results["land_incorrect"]

        print(f"\n=== SUMMARY ===")
        print(f"Water points: {results['water_correct']}/{total_water} correct")
        print(f"Land points: {results['land_correct']}/{total_land} correct")

    except Exception as e:
        print(f"Validation error: {e}")
