#!/usr/bin/env python3
"""Generate mangrove polygon based on elevation threshold discovery.

Mangroves exist in the elevation band between mean sea level (0m) and high tide (1.1m).
This was discovered empirically - the 0m threshold aligns with biological requirements:
- Below 0m: flooded >50% of time, too wet for mangrove establishment
- Above 0m: exposed >50% of time, allowing root respiration and seedling anchoring

This script generates a GeoJSON polygon of the mangrove zone.
"""

import json
import numpy as np
from pathlib import Path
from pyproj import Transformer
import matplotlib.pyplot as plt
import contextily as ctx
from skimage import measure
from scipy import ndimage

from waitangi.data.elevation import (
    compute_flooded_area_at_level,
    fetch_waitangi_elevation,
)
from waitangi.data.reference_points import (
    WATER_POINTS_WEST,
    WATER_POINTS_EAST,
    MANGROVE_POINTS,
    LAND_POINTS,
    LANDMARKS,
)


# Mangrove elevation thresholds (discovered empirically)
MANGROVE_LOWER_THRESHOLD = 0.0   # Mean sea level - below this is mudflat/sandbar
MANGROVE_UPPER_THRESHOLD = 1.1  # High tide level - above this is dry land


def latlon_to_nztm(lat: float, lon: float) -> tuple[float, float]:
    """Convert lat/lon to NZTM coordinates."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
    return transformer.transform(lon, lat)


def nztm_to_latlon(easting: float, northing: float) -> tuple[float, float]:
    """Convert NZTM to lat/lon coordinates."""
    transformer = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(easting, northing)
    return lat, lon


def pixel_to_nztm(row: int, col: int, transform) -> tuple[float, float]:
    """Convert pixel coordinates to NZTM."""
    easting = transform.c + col * transform.a
    northing = transform.f + row * transform.e
    return easting, northing


def get_extent_nztm(elev_data) -> list[float]:
    """Get the extent of elevation data in NZTM coordinates."""
    return [elev_data.bounds[0], elev_data.bounds[2],
            elev_data.bounds[1], elev_data.bounds[3]]


def plot_reference_points_nztm(ax, elev_data):
    """Plot reference points on axis."""
    point_groups = [
        (WATER_POINTS_WEST + WATER_POINTS_EAST, "cyan", "o", "Water", 20),
        (MANGROVE_POINTS, "lime", "^", "Mangrove (verified)", 25),
        (LAND_POINTS, "red", "s", "Land", 20),
        (LANDMARKS, "yellow", "*", "Landmarks", 40),
    ]

    for points, color, marker, label, size in point_groups:
        eastings, northings = [], []
        for pt in points:
            e, n = latlon_to_nztm(pt.lat, pt.lon)
            if (elev_data.bounds[0] <= e <= elev_data.bounds[2] and
                elev_data.bounds[1] <= n <= elev_data.bounds[3]):
                eastings.append(e)
                northings.append(n)

        if eastings:
            ax.scatter(eastings, northings, c=color, marker=marker, s=size,
                      label=label, edgecolors="black", linewidths=0.5, zorder=10)


def generate_mangrove_mask(elev_data, min_area_m2: float = 500) -> np.ndarray:
    """Generate boolean mask of mangrove zone.

    Mangroves exist where:
    - Elevation is above mean sea level (0m) - not permanently flooded
    - Elevation is below high tide (1.1m) - gets regular tidal flooding
    - Area is hydrologically connected to the estuary

    Args:
        elev_data: Elevation data
        min_area_m2: Minimum area in m² to keep (removes tiny fragments)
    """
    # Get flood masks at both thresholds
    _, lower_mask = compute_flooded_area_at_level(MANGROVE_LOWER_THRESHOLD, elev_data)
    _, upper_mask = compute_flooded_area_at_level(MANGROVE_UPPER_THRESHOLD, elev_data)

    # Mangrove zone: flooded at high tide but NOT at mean sea level
    mangrove_mask = upper_mask & ~lower_mask

    # Clean up the mask - remove tiny fragments
    pixel_area_m2 = abs(elev_data.transform.a * elev_data.transform.e)
    min_pixels = int(min_area_m2 / pixel_area_m2)

    # Label connected components and remove small ones
    labeled, num_features = ndimage.label(mangrove_mask)
    component_sizes = ndimage.sum(mangrove_mask, labeled, range(1, num_features + 1))

    # Keep only components larger than minimum size
    small_components = np.where(component_sizes < min_pixels)[0] + 1
    for comp_id in small_components:
        mangrove_mask[labeled == comp_id] = False

    # Also fill small holes inside the mangrove areas
    # Invert, remove small components, invert back
    inverted = ~mangrove_mask
    labeled_holes, num_holes = ndimage.label(inverted)
    hole_sizes = ndimage.sum(inverted, labeled_holes, range(1, num_holes + 1))

    # Fill holes smaller than threshold (but not the main water body)
    small_holes = np.where(hole_sizes < min_pixels * 2)[0] + 1
    for hole_id in small_holes:
        # Only fill if it's surrounded by mangrove (not touching edge)
        hole_mask = labeled_holes == hole_id
        if not (hole_mask[0, :].any() or hole_mask[-1, :].any() or
                hole_mask[:, 0].any() or hole_mask[:, -1].any()):
            mangrove_mask[hole_mask] = True

    return mangrove_mask


def mask_to_polygons_nztm(mask: np.ndarray, transform) -> list[list[tuple[float, float]]]:
    """Convert boolean mask to list of polygon coordinates in NZTM."""
    # Find contours at 0.5 level (boundary of True/False)
    contours = measure.find_contours(mask.astype(float), 0.5)

    polygons = []
    for contour in contours:
        # Convert pixel coords to NZTM
        # contour is in (row, col) format
        coords = []
        for row, col in contour:
            easting, northing = pixel_to_nztm(row, col, transform)
            coords.append((easting, northing))

        # Close the polygon
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])

        # Only include polygons with reasonable size (>10 points)
        if len(coords) > 10:
            polygons.append(coords)

    return polygons


def mask_to_geojson(mask: np.ndarray, transform) -> dict:
    """Convert boolean mask to GeoJSON in WGS84."""
    contours = measure.find_contours(mask.astype(float), 0.5)

    features = []
    for i, contour in enumerate(contours):
        # Skip tiny polygons
        if len(contour) < 10:
            continue

        # Convert to lat/lon
        coords = []
        for row, col in contour:
            easting, northing = pixel_to_nztm(row, col, transform)
            lat, lon = nztm_to_latlon(easting, northing)
            coords.append([lon, lat])

        # Close polygon
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])

        # Calculate approximate area
        pixel_area = abs(transform.a * transform.e)
        # Count pixels inside this contour (approximate)

        feature = {
            "type": "Feature",
            "properties": {
                "id": i,
                "zone": "mangrove",
                "lower_threshold_m": MANGROVE_LOWER_THRESHOLD,
                "upper_threshold_m": MANGROVE_UPPER_THRESHOLD,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            }
        }
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def main():
    print("Fetching elevation data...")
    elev = fetch_waitangi_elevation()

    print(f"\nMangrove elevation band: {MANGROVE_LOWER_THRESHOLD}m to {MANGROVE_UPPER_THRESHOLD}m")
    print("  - Lower threshold (0m): Mean sea level - below is mudflat/sandbar")
    print("  - Upper threshold (1.1m): High tide - above is dry land")

    print("\nGenerating mangrove mask...")
    mangrove_mask = generate_mangrove_mask(elev)

    # Calculate area
    pixel_area_m2 = abs(elev.transform.a * elev.transform.e)
    mangrove_area_km2 = np.sum(mangrove_mask) * pixel_area_m2 / 1e6
    print(f"Mangrove zone area: {mangrove_area_km2:.3f} km²")

    # Generate GeoJSON
    print("\nGenerating GeoJSON...")
    geojson = mask_to_geojson(mangrove_mask, elev.transform)

    geojson_path = Path("waitangi_mangroves.geojson")
    with open(geojson_path, "w") as f:
        json.dump(geojson, f, indent=2)
    print(f"Saved GeoJSON to: {geojson_path}")
    print(f"  - {len(geojson['features'])} polygon(s)")

    # Create visualization
    print("\nGenerating visualization...")
    fig, ax = plt.subplots(figsize=(14, 12))

    extent_nztm = get_extent_nztm(elev)
    img_extent = [extent_nztm[0], extent_nztm[1], extent_nztm[2], extent_nztm[3]]

    ax.set_xlim(extent_nztm[0], extent_nztm[1])
    ax.set_ylim(extent_nztm[2], extent_nztm[3])

    # Satellite basemap
    try:
        ctx.add_basemap(ax, crs="EPSG:2193", source=ctx.providers.Esri.WorldImagery, zoom=15)
    except Exception as e:
        print(f"Could not load basemap: {e}")

    # Get water mask at mean sea level for context
    _, water_mask = compute_flooded_area_at_level(MANGROVE_LOWER_THRESHOLD, elev)

    # Show water (blue)
    ax.imshow(np.where(water_mask, 1, np.nan), cmap="Blues", origin="upper",
              extent=img_extent, alpha=0.5, vmin=0, vmax=2)

    # Show mangrove zone (green)
    ax.imshow(np.where(mangrove_mask, 1, np.nan), cmap="Greens", origin="upper",
              extent=img_extent, alpha=0.6, vmin=0, vmax=2)

    # Plot reference points
    plot_reference_points_nztm(ax, elev)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", alpha=0.5, label="Water (at 0m / mean sea level)"),
        Patch(facecolor="green", alpha=0.6, label=f"Mangrove zone ({MANGROVE_LOWER_THRESHOLD}m to {MANGROVE_UPPER_THRESHOLD}m)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.9)

    ax.set_title(
        f"Waitangi Estuary - Mangrove Zone\n"
        f"Elevation band: {MANGROVE_LOWER_THRESHOLD}m (mean sea level) to {MANGROVE_UPPER_THRESHOLD}m (high tide)\n"
        f"Total mangrove area: {mangrove_area_km2:.3f} km²",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    plt.tight_layout()

    output_path = "waitangi_mangroves.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to: {output_path}")
    plt.close()

    # Validate against known mangrove points
    print("\n=== Validation against known mangrove points ===")
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)

    hits = 0
    misses = 0
    for pt in MANGROVE_POINTS:
        easting, northing = transformer.transform(pt.lon, pt.lat)
        col = int((easting - elev.transform.c) / elev.transform.a)
        row = int((northing - elev.transform.f) / elev.transform.e)

        if 0 <= row < mangrove_mask.shape[0] and 0 <= col < mangrove_mask.shape[1]:
            if mangrove_mask[row, col]:
                hits += 1
            else:
                misses += 1
                elevation = elev.data[row, col]
                print(f"  MISS: {pt.name} (elevation: {elevation:.2f}m)")
        else:
            print(f"  OUT OF BOUNDS: {pt.name}")

    print(f"\nMangrove point validation: {hits}/{hits+misses} points inside predicted mangrove zone")


if __name__ == "__main__":
    main()
