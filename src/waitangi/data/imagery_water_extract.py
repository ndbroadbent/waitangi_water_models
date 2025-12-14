"""Extract water boundaries from LINZ aerial imagery using color analysis.

Uses LINZ Basemaps XYZ tile API to fetch high-resolution aerial imagery,
then applies color-based segmentation to identify water areas.

Water colors observed in Waitangi estuary:
- 948663, 426552, 73755a, 446a55 (muddy tidal water)
- Boundaries near 202622 => 415865

Land colors (surrounding bush):
- 334854 (trees/vegetation)
"""

import io
import math
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import numpy as np
from PIL import Image
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union

from waitangi.core.config import WaitangiLocation, get_settings

if TYPE_CHECKING:
    from numpy.typing import NDArray


# LINZ Basemaps XYZ tile URL template
# Uses EPSG:3857 (Web Mercator) for XYZ tiles
LINZ_BASEMAP_URL = "https://basemaps.linz.govt.nz/v1/tiles/aerial/3857/{z}/{x}/{y}.webp"

# Tile size in pixels
TILE_SIZE = 256

# Zoom level for water extraction (higher = more detail but more tiles)
# Zoom 18 = ~0.6m/pixel, Zoom 17 = ~1.2m/pixel, Zoom 16 = ~2.4m/pixel
DEFAULT_ZOOM = 17


def _get_linz_api_key() -> str:
    """Get LINZ API key from settings."""
    settings = get_settings()
    key = settings.data_sources.linz_api_key
    if not key:
        raise ValueError("LINZ API key required for basemap tiles")
    return key


def _lat_lon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Convert lat/lon to XYZ tile coordinates."""
    lat_rad = math.radians(lat)
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _tile_to_lat_lon(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
    """Convert tile coords to bounding box (min_lat, min_lon, max_lat, max_lon)."""
    n = 2**zoom

    # Top-left corner
    lon_min = x / n * 360.0 - 180.0
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))

    # Bottom-right corner
    lon_max = (x + 1) / n * 360.0 - 180.0
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))

    return lat_min, lon_min, lat_max, lon_max


def fetch_tile(x: int, y: int, zoom: int) -> Image.Image:
    """Fetch a single tile from LINZ basemaps."""
    api_key = _get_linz_api_key()
    url = f"{LINZ_BASEMAP_URL}?api={api_key}".format(z=zoom, x=x, y=y)

    with httpx.Client(timeout=30.0) as client:
        response = client.get(url)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))


def fetch_tiles_for_bbox(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    zoom: int = DEFAULT_ZOOM,
    cache_dir: Path | None = None,
) -> tuple[Image.Image, tuple[float, float, float, float]]:
    """Fetch and stitch tiles covering a bounding box.

    Args:
        min_lat, min_lon, max_lat, max_lon: Bounding box in WGS84
        zoom: Tile zoom level
        cache_dir: Optional directory to cache tiles

    Returns:
        Tuple of (stitched image, actual bbox covered)
    """
    api_key = _get_linz_api_key()

    # Get tile range
    x_min, y_max = _lat_lon_to_tile(min_lat, min_lon, zoom)
    x_max, y_min = _lat_lon_to_tile(max_lat, max_lon, zoom)

    # Ensure correct ordering
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    # Calculate output image size
    width = (x_max - x_min + 1) * TILE_SIZE
    height = (y_max - y_min + 1) * TILE_SIZE

    # Create output image
    result = Image.new("RGB", (width, height))

    # Set up cache
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    # Fetch and stitch tiles
    with httpx.Client(timeout=30.0) as client:
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                # Check cache first
                cache_path = cache_dir / f"z{zoom}_x{x}_y{y}.webp" if cache_dir else None
                if cache_path and cache_path.exists():
                    tile = Image.open(cache_path)
                else:
                    url = LINZ_BASEMAP_URL.format(z=zoom, x=x, y=y) + f"?api={api_key}"
                    response = client.get(url)
                    response.raise_for_status()
                    tile = Image.open(io.BytesIO(response.content))

                    # Cache the tile
                    if cache_path:
                        tile.save(cache_path, "WEBP")

                # Convert to RGB if needed
                if tile.mode != "RGB":
                    tile = tile.convert("RGB")

                # Paste into result
                px = (x - x_min) * TILE_SIZE
                py = (y - y_min) * TILE_SIZE
                result.paste(tile, (px, py))

    # Calculate actual bbox covered
    actual_bbox = (
        _tile_to_lat_lon(x_min, y_max + 1, zoom)[0],  # min_lat
        _tile_to_lat_lon(x_min, y_min, zoom)[1],  # min_lon
        _tile_to_lat_lon(x_max + 1, y_min, zoom)[2],  # max_lat
        _tile_to_lat_lon(x_max, y_max, zoom)[3],  # max_lon
    )

    return result, actual_bbox


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hsv(r: int, g: int, b: int) -> tuple[float, float, float]:
    """Convert RGB to HSV (0-360, 0-100, 0-100)."""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn

    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    else:
        h = (60 * ((r - g) / df) + 240) % 360

    s = 0 if mx == 0 else (df / mx) * 100
    v = mx * 100

    return h, s, v


def detect_water_mask(
    image: Image.Image,
    water_colors: list[str] | None = None,
    tolerance: int = 35,
) -> "NDArray[np.bool_]":
    """Detect water pixels in an image using color analysis.

    Args:
        image: Input RGB image
        water_colors: List of hex colors representing water (default: Waitangi estuary colors)
        tolerance: Color distance tolerance for matching

    Returns:
        Boolean mask where True = water
    """
    if water_colors is None:
        # Waitangi estuary tidal water colors
        water_colors = [
            "948663",  # Muddy brown-green
            "426552",  # Dark teal
            "73755a",  # Olive-brown
            "446a55",  # Dark green-teal
            "415865",  # Blue-grey
            "5a7065",  # Grey-green
            "6b8070",  # Lighter grey-green
            "4a6055",  # Mid teal
        ]

    # Convert image to numpy array
    img_array = np.array(image)

    # Convert water colors to RGB
    water_rgb = [_hex_to_rgb(c) for c in water_colors]

    # Create mask
    mask = np.zeros(img_array.shape[:2], dtype=bool)

    # Check each water color
    for wr, wg, wb in water_rgb:
        # Calculate color distance
        dist = np.sqrt(
            (img_array[:, :, 0].astype(float) - wr) ** 2
            + (img_array[:, :, 1].astype(float) - wg) ** 2
            + (img_array[:, :, 2].astype(float) - wb) ** 2
        )
        mask |= dist < tolerance

    # Also detect by HSV characteristics of muddy water
    # Muddy estuary water tends to have:
    # - Hue: 60-180 (green-cyan range)
    # - Saturation: 10-50 (not too saturated)
    # - Value: 30-70 (mid brightness)
    for y in range(img_array.shape[0]):
        for x in range(img_array.shape[1]):
            if mask[y, x]:
                continue
            r, g, b = img_array[y, x]
            h, s, v = _rgb_to_hsv(r, g, b)
            # Check if it looks like muddy water
            if 40 <= h <= 200 and 10 <= s <= 60 and 25 <= v <= 55:
                mask[y, x] = True

    return mask


def mask_to_polygon(
    mask: "NDArray[np.bool_]",
    bbox: tuple[float, float, float, float],
    simplify_tolerance: float = 0.00001,
) -> Polygon | MultiPolygon:
    """Convert a boolean mask to a polygon in WGS84 coordinates.

    Args:
        mask: Boolean mask (True = water)
        bbox: (min_lat, min_lon, max_lat, max_lon) of the mask
        simplify_tolerance: Simplification tolerance in degrees

    Returns:
        Polygon or MultiPolygon in WGS84 coordinates
    """
    from rasterio.features import shapes
    from shapely.geometry import shape as shapely_shape

    min_lat, min_lon, max_lat, max_lon = bbox
    height, width = mask.shape

    # Calculate pixel size in degrees
    pixel_width = (max_lon - min_lon) / width
    pixel_height = (max_lat - min_lat) / height

    # Extract polygons from mask
    polygons = []
    for geom, value in shapes(mask.astype(np.uint8), mask=mask):
        if value == 1:
            poly = shapely_shape(geom)

            # Transform pixel coordinates to lat/lon
            def transform_coords(coords):
                return [
                    (min_lon + x * pixel_width, max_lat - y * pixel_height)
                    for x, y in coords
                ]

            if poly.geom_type == "Polygon":
                exterior = transform_coords(poly.exterior.coords)
                interiors = [transform_coords(ring.coords) for ring in poly.interiors]
                poly = Polygon(exterior, interiors)
            elif poly.geom_type == "MultiPolygon":
                new_polys = []
                for p in poly.geoms:
                    exterior = transform_coords(p.exterior.coords)
                    interiors = [transform_coords(ring.coords) for ring in p.interiors]
                    new_polys.append(Polygon(exterior, interiors))
                poly = MultiPolygon(new_polys)

            if poly.is_valid and poly.area > 0:
                polygons.append(poly)

    if not polygons:
        return Polygon()

    # Merge all polygons
    result = unary_union(polygons)

    # Simplify
    if simplify_tolerance > 0:
        result = result.simplify(simplify_tolerance)

    return result


@lru_cache(maxsize=1)
def extract_water_polygon_from_imagery(
    min_lat: float = -35.285649,
    min_lon: float = 174.049555,
    max_lat: float = -35.262348,
    max_lon: float = 174.085256,
    zoom: int = DEFAULT_ZOOM,
) -> Polygon | MultiPolygon:
    """Extract water polygon from LINZ aerial imagery.

    Args:
        min_lat, min_lon, max_lat, max_lon: Bounding box
        zoom: Tile zoom level

    Returns:
        Water polygon in WGS84 coordinates
    """
    settings = get_settings()
    cache_dir = settings.data_sources.cache_dir / "linz_tiles"

    print(f"Fetching aerial imagery for bbox ({min_lat}, {min_lon}) to ({max_lat}, {max_lon})...")
    image, actual_bbox = fetch_tiles_for_bbox(
        min_lat, min_lon, max_lat, max_lon, zoom=zoom, cache_dir=cache_dir
    )

    print(f"Image size: {image.size}, detecting water...")
    mask = detect_water_mask(image)

    water_pct = mask.sum() / mask.size * 100
    print(f"Water coverage: {water_pct:.1f}%")

    print("Converting mask to polygon...")
    polygon = mask_to_polygon(mask, actual_bbox)

    print(f"Result: {polygon.geom_type} with area {polygon.area:.6f} deg²")
    return polygon


def extract_water_polygon_nztm(
    min_lat: float = -35.285649,
    min_lon: float = 174.049555,
    max_lat: float = -35.262348,
    max_lon: float = 174.085256,
    zoom: int = DEFAULT_ZOOM,
) -> Polygon | MultiPolygon:
    """Extract water polygon and convert to NZTM coordinates."""
    from shapely.ops import transform

    polygon_wgs84 = extract_water_polygon_from_imagery(min_lat, min_lon, max_lat, max_lon, zoom)

    # Transform to NZTM
    transformer = Transformer.from_crs(
        WaitangiLocation.CRS_WGS84, WaitangiLocation.CRS_NZTM, always_xy=True
    )

    def project(x, y):
        return transformer.transform(x, y)

    return transform(project, polygon_wgs84)


if __name__ == "__main__":
    # Test extraction
    print("=== LINZ IMAGERY WATER EXTRACTION ===\n")

    # User's visualization bbox
    polygon = extract_water_polygon_from_imagery()

    print(f"\nResult geometry type: {polygon.geom_type}")
    if not polygon.is_empty:
        bounds = polygon.bounds
        print(f"Bounds: {bounds}")
