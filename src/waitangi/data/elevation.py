"""LINZ LiDAR DEM elevation data for Waitangi estuary bathymetry.

Fetches 1m resolution elevation data from LINZ Cloud-Optimized GeoTIFFs
to model water volume at different tide levels.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds

from waitangi.core.config import WaitangiLocation
from waitangi.data.estuary_geometry import VISUALIZATION_BBOX_WGS84

if TYPE_CHECKING:
    from numpy.typing import NDArray

# LINZ LiDAR DEM Cloud-Optimized GeoTIFF URLs
# National composite DEM at 1m resolution
LINZ_DEM_BASE_URL = (
    "https://nz-elevation.s3.ap-southeast-2.amazonaws.com/"
    "new-zealand/new-zealand/dem_1m/2193"
)

# Waitangi area falls within AV29 sheet (northern portion)
WAITANGI_DEM_SHEET = "AV29"


@dataclass
class ElevationData:
    """Container for elevation raster data."""

    data: "NDArray[np.float32]"  # Elevation values in meters
    transform: rasterio.Affine  # Affine transform for coordinates
    crs: str  # Coordinate reference system (EPSG:2193)
    bounds: tuple[float, float, float, float]  # (west, south, east, north) in NZTM
    resolution: float  # Cell size in meters
    nodata: float | None  # NoData value


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


def get_dem_url(sheet: str = WAITANGI_DEM_SHEET) -> str:
    """Get URL for a LINZ DEM sheet."""
    return f"{LINZ_DEM_BASE_URL}/{sheet}.tiff"


@lru_cache(maxsize=1)
def fetch_waitangi_elevation() -> ElevationData:
    """Fetch elevation data for the Waitangi visualization area.

    Uses Cloud-Optimized GeoTIFF to efficiently fetch only the required
    subset of the 1m resolution LiDAR DEM.

    Returns:
        ElevationData with elevation values in meters relative to NZVD2016.
        Negative values represent areas below mean sea level.
    """
    transformer = _get_transformer_to_nztm()

    # Convert visualization bbox to NZTM
    min_lon, min_lat, max_lon, max_lat = VISUALIZATION_BBOX_WGS84
    min_e, min_n = transformer.transform(min_lon, min_lat)
    max_e, max_n = transformer.transform(max_lon, max_lat)

    cog_url = get_dem_url()

    with rasterio.open(cog_url) as src:
        # Create window for our area of interest
        window = from_bounds(min_e, min_n, max_e, max_n, src.transform)

        # Read the data
        data = src.read(1, window=window)

        # Get the transform for the windowed data
        win_transform = src.window_transform(window)

        # Calculate actual bounds from window
        west = win_transform.c
        north = win_transform.f
        east = west + data.shape[1] * win_transform.a
        south = north + data.shape[0] * win_transform.e

        return ElevationData(
            data=data.astype(np.float32),
            transform=win_transform,
            crs=str(src.crs),
            bounds=(west, south, east, north),
            resolution=abs(win_transform.a),
            nodata=src.nodata,
        )


def get_elevation_at_point(
    easting: float, northing: float, elevation_data: ElevationData | None = None
) -> float:
    """Get elevation at a specific NZTM point.

    Args:
        easting: NZTM easting coordinate
        northing: NZTM northing coordinate
        elevation_data: Pre-fetched elevation data (fetched if not provided)

    Returns:
        Elevation in meters (NZVD2016 datum)
    """
    if elevation_data is None:
        elevation_data = fetch_waitangi_elevation()

    # Convert coordinate to pixel indices
    col = int((easting - elevation_data.transform.c) / elevation_data.transform.a)
    row = int((northing - elevation_data.transform.f) / elevation_data.transform.e)

    # Check bounds
    if 0 <= row < elevation_data.data.shape[0] and 0 <= col < elevation_data.data.shape[1]:
        return float(elevation_data.data[row, col])

    return float("nan")


def get_elevation_at_wgs84(
    lat: float, lon: float, elevation_data: ElevationData | None = None
) -> float:
    """Get elevation at a WGS84 coordinate.

    Args:
        lat: Latitude (decimal degrees)
        lon: Longitude (decimal degrees)
        elevation_data: Pre-fetched elevation data (fetched if not provided)

    Returns:
        Elevation in meters (NZVD2016 datum)
    """
    transformer = _get_transformer_to_nztm()
    easting, northing = transformer.transform(lon, lat)
    return get_elevation_at_point(easting, northing, elevation_data)


def _flood_fill_from_seed(
    elevation: "NDArray[np.float32]",
    water_level: float,
    seed_row: int,
    seed_col: int,
) -> "NDArray[np.bool_]":
    """Flood fill from a seed point, only flooding connected cells below water level.

    Uses BFS to find all cells that are:
    1. Below the water level
    2. Connected (8-way) to the seed point through other below-water cells

    This ensures we only flood areas hydrologically connected to the ocean,
    not isolated low-lying inland areas.
    """
    from collections import deque

    rows, cols = elevation.shape
    flooded = np.zeros((rows, cols), dtype=bool)

    # Check seed is valid and below water
    if not (0 <= seed_row < rows and 0 <= seed_col < cols):
        return flooded
    if elevation[seed_row, seed_col] >= water_level:
        return flooded

    # BFS flood fill
    queue = deque([(seed_row, seed_col)])
    flooded[seed_row, seed_col] = True

    # 8-way connectivity
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while queue:
        r, c = queue.popleft()

        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols:
                if not flooded[nr, nc] and elevation[nr, nc] < water_level:
                    flooded[nr, nc] = True
                    queue.append((nr, nc))

    return flooded


def get_ocean_seed_point(elevation_data: ElevationData) -> tuple[int, int]:
    """Get a seed point in the ocean/estuary for flood filling.

    Returns (row, col) of a known DEEP water point in the estuary.
    Must be a point that's underwater even at low tide.
    """
    transformer = _get_transformer_to_nztm()

    # Use "West of bridge 3" which has elevation -1.30m (always underwater)
    # This is in the main channel, not near shore
    water_lon, water_lat = 174.071454, -35.273630

    easting, northing = transformer.transform(water_lon, water_lat)

    # Convert to pixel coordinates
    col = int((easting - elevation_data.transform.c) / elevation_data.transform.a)
    row = int((northing - elevation_data.transform.f) / elevation_data.transform.e)

    return row, col


def compute_flooded_area_at_level(
    water_level: float, elevation_data: ElevationData | None = None
) -> tuple[float, "NDArray[np.bool_]"]:
    """Compute flooded area for a given water level using hydrological connectivity.

    Only floods areas that are connected to the ocean/estuary, not isolated
    low-lying inland areas.

    Args:
        water_level: Water surface elevation in meters (NZVD2016)
        elevation_data: Pre-fetched elevation data (fetched if not provided)

    Returns:
        Tuple of (flooded_area_m2, flooded_mask)
    """
    if elevation_data is None:
        elevation_data = fetch_waitangi_elevation()

    # Get seed point in the ocean
    seed_row, seed_col = get_ocean_seed_point(elevation_data)

    # Flood fill from ocean, respecting elevation
    flooded = _flood_fill_from_seed(
        elevation_data.data, water_level, seed_row, seed_col
    )

    # Calculate area (each cell is resolution x resolution)
    cell_area = elevation_data.resolution**2
    flooded_area = float(np.sum(flooded)) * cell_area

    return flooded_area, flooded


def compute_water_volume_at_level(
    water_level: float, elevation_data: ElevationData | None = None
) -> tuple[float, float, "NDArray[np.bool_]"]:
    """Compute water volume for a given water level using hydrological connectivity.

    Only counts volume in areas connected to the ocean/estuary.

    Args:
        water_level: Water surface elevation in meters (NZVD2016)
        elevation_data: Pre-fetched elevation data (fetched if not provided)

    Returns:
        Tuple of (volume_m3, flooded_area_m2, flooded_mask)
    """
    if elevation_data is None:
        elevation_data = fetch_waitangi_elevation()

    # Get flood-filled mask (only hydrologically connected areas)
    flooded_area, flooded = compute_flooded_area_at_level(water_level, elevation_data)

    # Calculate depth at each flooded cell
    depths = np.where(flooded, water_level - elevation_data.data, 0)

    # Calculate volume (depth * cell_area)
    cell_area = elevation_data.resolution**2
    volume = float(np.sum(depths)) * cell_area

    return volume, flooded_area, flooded


def compute_tidal_prism(
    low_tide_level: float,
    high_tide_level: float,
    elevation_data: ElevationData | None = None,
) -> dict[str, float]:
    """Compute tidal prism (volume change between low and high tide).

    The tidal prism is the volume of water exchanged between low and high tide,
    which drives tidal currents through constricted channels.

    Args:
        low_tide_level: Low tide water level (meters, NZVD2016)
        high_tide_level: High tide water level (meters, NZVD2016)
        elevation_data: Pre-fetched elevation data (fetched if not provided)

    Returns:
        Dictionary with:
        - tidal_prism_m3: Volume difference
        - low_tide_area_m2: Flooded area at low tide
        - high_tide_area_m2: Flooded area at high tide
        - area_change_m2: Area difference
        - avg_depth_change_m: Average depth change over high tide area
    """
    if elevation_data is None:
        elevation_data = fetch_waitangi_elevation()

    vol_low, area_low, _ = compute_water_volume_at_level(low_tide_level, elevation_data)
    vol_high, area_high, _ = compute_water_volume_at_level(high_tide_level, elevation_data)

    tidal_prism = vol_high - vol_low
    area_change = area_high - area_low

    # Average depth change over the high tide area
    avg_depth_change = tidal_prism / area_high if area_high > 0 else 0

    return {
        "tidal_prism_m3": tidal_prism,
        "low_tide_area_m2": area_low,
        "high_tide_area_m2": area_high,
        "area_change_m2": area_change,
        "avg_depth_change_m": avg_depth_change,
        "low_tide_volume_m3": vol_low,
        "high_tide_volume_m3": vol_high,
    }


def get_elevation_stats(elevation_data: ElevationData | None = None) -> dict[str, float]:
    """Get basic statistics about the elevation data.

    Returns:
        Dictionary with min, max, mean elevation and coverage info.
    """
    if elevation_data is None:
        elevation_data = fetch_waitangi_elevation()

    data = elevation_data.data

    # Handle nodata
    if elevation_data.nodata is not None:
        valid_mask = data != elevation_data.nodata
        valid_data = data[valid_mask]
    else:
        valid_data = data.flatten()

    return {
        "min_elevation_m": float(np.min(valid_data)),
        "max_elevation_m": float(np.max(valid_data)),
        "mean_elevation_m": float(np.mean(valid_data)),
        "std_elevation_m": float(np.std(valid_data)),
        "rows": elevation_data.data.shape[0],
        "cols": elevation_data.data.shape[1],
        "resolution_m": elevation_data.resolution,
        "bounds_nztm": elevation_data.bounds,
    }


def find_channel_constrictions(
    water_level: float,
    elevation_data: ElevationData | None = None,
    min_width_m: float = 5.0,
    max_width_m: float = 100.0,
) -> list[dict]:
    """Find narrow channel constrictions where flow velocities would be highest.

    Scans across the flooded area to find narrow cross-sections where
    water must squeeze through, creating faster currents.

    Args:
        water_level: Water surface elevation (meters, NZVD2016)
        elevation_data: Pre-fetched elevation data
        min_width_m: Minimum channel width to consider
        max_width_m: Maximum width to count as a "constriction"

    Returns:
        List of constriction dictionaries with location and width info.
    """
    if elevation_data is None:
        elevation_data = fetch_waitangi_elevation()

    # Create flooded mask
    flooded = elevation_data.data < water_level
    if elevation_data.nodata is not None:
        valid = elevation_data.data != elevation_data.nodata
        flooded = flooded & valid

    constrictions = []
    res = elevation_data.resolution

    # Scan east-west lines (measuring north-south channel widths)
    for col in range(0, flooded.shape[1], 10):  # Every 10 meters
        water_segments = []
        in_water = False
        start_row = 0

        for row in range(flooded.shape[0]):
            if flooded[row, col] and not in_water:
                in_water = True
                start_row = row
            elif not flooded[row, col] and in_water:
                in_water = False
                width = (row - start_row) * res
                if min_width_m <= width <= max_width_m:
                    center_row = (start_row + row) // 2
                    water_segments.append({
                        "row": center_row,
                        "col": col,
                        "width_m": width,
                        "direction": "NS",
                    })

        constrictions.extend(water_segments)

    # Scan north-south lines (measuring east-west channel widths)
    for row in range(0, flooded.shape[0], 10):  # Every 10 meters
        water_segments = []
        in_water = False
        start_col = 0

        for col in range(flooded.shape[1]):
            if flooded[row, col] and not in_water:
                in_water = True
                start_col = col
            elif not flooded[row, col] and in_water:
                in_water = False
                width = (col - start_col) * res
                if min_width_m <= width <= max_width_m:
                    center_col = (start_col + col) // 2
                    water_segments.append({
                        "row": row,
                        "col": center_col,
                        "width_m": width,
                        "direction": "EW",
                    })

        constrictions.extend(water_segments)

    # Convert pixel coordinates to NZTM and WGS84
    transformer = _get_transformer_to_wgs84()

    for c in constrictions:
        # Pixel to NZTM
        easting = elevation_data.transform.c + c["col"] * elevation_data.transform.a
        northing = elevation_data.transform.f + c["row"] * elevation_data.transform.e
        c["easting"] = easting
        c["northing"] = northing

        # NZTM to WGS84
        lon, lat = transformer.transform(easting, northing)
        c["lat"] = lat
        c["lon"] = lon

    # Sort by width (narrowest first - these are the most interesting for kayaking)
    constrictions.sort(key=lambda x: x["width_m"])

    return constrictions


def estimate_channel_velocity(
    channel_width_m: float,
    channel_depth_m: float,
    tidal_prism_m3: float,
    tidal_period_s: float = 6 * 3600,  # ~6 hours for ebb or flood
) -> float:
    """Estimate peak flow velocity through a channel constriction.

    Uses simplified continuity equation: Q = A * v
    where Q is volumetric flow rate, A is cross-sectional area, v is velocity.

    Assumes all tidal prism flows through this channel, so this gives
    a maximum theoretical velocity (actual will be less if multiple channels).

    Args:
        channel_width_m: Width of the channel
        channel_depth_m: Average depth of the channel
        tidal_prism_m3: Volume of water that must flow through
        tidal_period_s: Time for half-tide cycle (default 6 hours)

    Returns:
        Estimated peak velocity in m/s
    """
    # Cross-sectional area
    area = channel_width_m * channel_depth_m

    # Average flow rate (volume / time)
    # Peak flow is roughly π/2 times average for sinusoidal tide
    avg_flow_rate = tidal_prism_m3 / tidal_period_s
    peak_flow_rate = avg_flow_rate * (np.pi / 2)

    # Velocity = flow rate / area
    if area > 0:
        velocity = peak_flow_rate / area
    else:
        velocity = 0.0

    return float(velocity)


def analyze_kayak_channels(
    elevation_data: ElevationData | None = None,
    low_tide: float = -0.5,
    high_tide: float = 1.5,
) -> list[dict]:
    """Analyze channels for kayak-friendly fast currents.

    Identifies narrow channels and estimates flow velocities
    that would be fun for kayaking.

    Args:
        elevation_data: Pre-fetched elevation data
        low_tide: Low tide level (meters)
        high_tide: High tide level (meters)

    Returns:
        List of channel analyses sorted by estimated velocity.
    """
    if elevation_data is None:
        elevation_data = fetch_waitangi_elevation()

    # Get tidal prism
    prism = compute_tidal_prism(low_tide, high_tide, elevation_data)

    # Find constrictions at mid-tide
    mid_tide = (low_tide + high_tide) / 2
    constrictions = find_channel_constrictions(mid_tide, elevation_data)

    # Filter and analyze the narrowest channels
    channels = []
    for c in constrictions[:50]:  # Top 50 narrowest
        # Estimate average depth (difference between water level and bed)
        elev_at_point = get_elevation_at_point(c["easting"], c["northing"], elevation_data)
        depth = mid_tide - elev_at_point

        if depth < 0.3:  # Too shallow for kayaking
            continue

        # Estimate velocity
        velocity = estimate_channel_velocity(
            channel_width_m=c["width_m"],
            channel_depth_m=depth,
            tidal_prism_m3=prism["tidal_prism_m3"],
        )

        channels.append({
            "lat": c["lat"],
            "lon": c["lon"],
            "width_m": c["width_m"],
            "depth_m": depth,
            "velocity_ms": velocity,
            "velocity_knots": velocity * 1.944,  # Convert to knots
            "direction": c["direction"],
        })

    # Sort by velocity (fastest first)
    channels.sort(key=lambda x: x["velocity_ms"], reverse=True)

    return channels


if __name__ == "__main__":
    print("=== LINZ LiDAR DEM for Waitangi ===\n")

    print("Fetching elevation data...")
    elev = fetch_waitangi_elevation()

    stats = get_elevation_stats(elev)
    print(f"\nElevation Statistics:")
    print(f"  Range: {stats['min_elevation_m']:.2f}m to {stats['max_elevation_m']:.2f}m")
    print(f"  Mean: {stats['mean_elevation_m']:.2f}m")
    print(f"  Resolution: {stats['resolution_m']}m")
    print(f"  Grid size: {stats['rows']} x {stats['cols']}")

    print("\n=== Tidal Analysis ===\n")

    # Typical tidal range for Bay of Islands
    low_tide = -0.5  # meters below NZVD2016
    high_tide = 1.5  # meters above NZVD2016

    prism = compute_tidal_prism(low_tide, high_tide, elev)

    print(f"Tide levels: {low_tide}m to {high_tide}m (range: {high_tide - low_tide}m)")
    print(f"\nTidal Prism: {prism['tidal_prism_m3']:,.0f} m³")
    print(f"  = {prism['tidal_prism_m3'] / 1e6:.3f} million m³")
    print(f"\nFlooded Area:")
    print(f"  Low tide: {prism['low_tide_area_m2'] / 1e6:.3f} km²")
    print(f"  High tide: {prism['high_tide_area_m2'] / 1e6:.3f} km²")
    print(f"  Change: {prism['area_change_m2'] / 1e4:.2f} hectares")
    print(f"\nAverage depth change: {prism['avg_depth_change_m']:.2f}m")

    print("\n=== Channel Analysis (Kayaking Spots) ===\n")

    print("Finding narrow channels...")
    channels = analyze_kayak_channels(elev, low_tide, high_tide)

    print(f"Found {len(channels)} potential kayak channels\n")

    print("Top 10 fastest channels (estimated):")
    for i, ch in enumerate(channels[:10], 1):
        print(f"  {i}. {ch['lat']:.5f}, {ch['lon']:.5f}")
        print(f"     Width: {ch['width_m']:.0f}m, Depth: {ch['depth_m']:.1f}m")
        print(f"     Est. velocity: {ch['velocity_ms']:.1f} m/s ({ch['velocity_knots']:.1f} knots)")
        print()
