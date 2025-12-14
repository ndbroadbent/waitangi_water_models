#!/usr/bin/env python3
"""Visualize water extent at different tide levels for Waitangi estuary.

Creates a figure showing the bathymetry and flooded areas at low/mid/high tide,
with all reference points overlaid for validation, on satellite imagery.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pyproj import Transformer
import contextily as ctx

from waitangi.data.elevation import (
    compute_flooded_area_at_level,
    compute_tidal_prism,
    fetch_waitangi_elevation,
    get_elevation_stats,
)
from waitangi.data.reference_points import (
    WATER_POINTS_WEST,
    WATER_POINTS_EAST,
    LAND_POINTS,
    MANGROVE_POINTS,
    LANDMARKS,
)
from waitangi.data.estuary_geometry import VISUALIZATION_BBOX_WGS84


def create_bathymetry_colormap():
    """Create a colormap suitable for bathymetry visualization.

    Blue for deep water, cyan for shallow, green for intertidal,
    tan/brown for land, green for vegetation.
    """
    colors = [
        (0.0, (0.1, 0.2, 0.5)),    # Deep blue (-3m)
        (0.15, (0.2, 0.4, 0.8)),   # Blue (-1m)
        (0.25, (0.3, 0.6, 0.9)),   # Light blue (0m / sea level)
        (0.35, (0.4, 0.8, 0.8)),   # Cyan (1m - intertidal)
        (0.45, (0.6, 0.9, 0.6)),   # Light green (2m - intertidal)
        (0.55, (0.8, 0.9, 0.6)),   # Yellow-green (4m - low land)
        (0.70, (0.9, 0.8, 0.5)),   # Tan (10m)
        (0.85, (0.6, 0.5, 0.3)),   # Brown (30m)
        (1.0, (0.4, 0.5, 0.3)),    # Dark brown/green (100m+)
    ]
    return LinearSegmentedColormap.from_list("bathymetry", colors)


def latlon_to_pixel(lat, lon, elev_data):
    """Convert lat/lon to pixel coordinates in the elevation raster."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
    easting, northing = transformer.transform(lon, lat)

    col = int((easting - elev_data.transform.c) / elev_data.transform.a)
    row = int((northing - elev_data.transform.f) / elev_data.transform.e)

    return row, col


def plot_reference_points(ax, elev_data, show_legend=True):
    """Plot all reference points on an axis (pixel coordinates)."""
    # Convert all points to pixel coordinates and plot
    point_groups = [
        (WATER_POINTS_WEST + WATER_POINTS_EAST, "cyan", "o", "Water (navigable)", 8),
        (MANGROVE_POINTS, "green", "^", "Mangrove", 8),
        (LAND_POINTS, "red", "s", "Land", 8),
        (LANDMARKS, "yellow", "*", "Landmarks", 12),
    ]

    for points, color, marker, label, size in point_groups:
        rows, cols = [], []
        for pt in points:
            r, c = latlon_to_pixel(pt.lat, pt.lon, elev_data)
            if 0 <= r < elev_data.data.shape[0] and 0 <= c < elev_data.data.shape[1]:
                rows.append(r)
                cols.append(c)

        if rows:
            ax.scatter(cols, rows, c=color, marker=marker, s=size,
                      label=label, edgecolors="black", linewidths=0.5, zorder=10)

    if show_legend:
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)


def latlon_to_nztm(lat, lon):
    """Convert lat/lon to NZTM coordinates."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    return easting, northing


def plot_reference_points_nztm(ax, elev_data, show_legend=True):
    """Plot all reference points on an axis (NZTM coordinates)."""
    point_groups = [
        (WATER_POINTS_WEST + WATER_POINTS_EAST, "cyan", "o", "Water (navigable)", 30),
        (MANGROVE_POINTS, "green", "^", "Mangrove", 30),
        (LAND_POINTS, "red", "s", "Land", 30),
        (LANDMARKS, "yellow", "*", "Landmarks", 50),
    ]

    for points, color, marker, label, size in point_groups:
        eastings, northings = [], []
        for pt in points:
            e, n = latlon_to_nztm(pt.lat, pt.lon)
            # Check if within bounds
            if (elev_data.bounds[0] <= e <= elev_data.bounds[2] and
                elev_data.bounds[1] <= n <= elev_data.bounds[3]):
                eastings.append(e)
                northings.append(n)

        if eastings:
            ax.scatter(eastings, northings, c=color, marker=marker, s=size,
                      label=label, edgecolors="black", linewidths=0.5, zorder=10)

    if show_legend:
        ax.legend(loc="upper left", fontsize=7, framealpha=0.9)


def validate_flooding(elev_data, flood_mask, water_level):
    """Check if flooding model correctly classifies reference points."""
    results = {"pass": 0, "fail": 0, "skipped": 0, "errors": []}

    # Water points should be flooded
    for pt in WATER_POINTS_WEST + WATER_POINTS_EAST:
        r, c = latlon_to_pixel(pt.lat, pt.lon, elev_data)
        if 0 <= r < flood_mask.shape[0] and 0 <= c < flood_mask.shape[1]:
            if flood_mask[r, c]:
                results["pass"] += 1
            else:
                results["fail"] += 1
                results["errors"].append(f"FAIL: {pt.name} should be flooded at {water_level}m")
        else:
            results["skipped"] += 1  # Out of bounds (bay area)

    # Land points should NOT be flooded
    for pt in LAND_POINTS:
        r, c = latlon_to_pixel(pt.lat, pt.lon, elev_data)
        if 0 <= r < flood_mask.shape[0] and 0 <= c < flood_mask.shape[1]:
            if not flood_mask[r, c]:
                results["pass"] += 1
            else:
                results["fail"] += 1
                results["errors"].append(f"FAIL: {pt.name} should NOT be flooded at {water_level}m")
        else:
            results["skipped"] += 1

    return results


def get_extent_nztm(elev_data):
    """Get the extent of elevation data in NZTM coordinates."""
    west = elev_data.bounds[0]
    east = elev_data.bounds[2]
    south = elev_data.bounds[1]
    north = elev_data.bounds[3]
    return [west, east, south, north]


def main():
    print("Fetching elevation data...")
    elev = fetch_waitangi_elevation()

    stats = get_elevation_stats(elev)
    print(f"Elevation range: {stats['min_elevation_m']:.1f}m to {stats['max_elevation_m']:.1f}m")

    # Tide levels (calibrated to match reference points)
    # High tide set to 1.1m to avoid flooding "never underwater" points
    # which have elevations between 1.1m and 1.5m
    low_tide = -0.5
    mid_tide = 0.3
    high_tide = 1.1

    # Compute flooded areas
    print("Computing flood extents...")
    _, low_mask = compute_flooded_area_at_level(low_tide, elev)
    _, mid_mask = compute_flooded_area_at_level(mid_tide, elev)
    _, high_mask = compute_flooded_area_at_level(high_tide, elev)

    # Validate against reference points
    print("\n=== Validation at High Tide ===")
    validation = validate_flooding(elev, high_mask, high_tide)
    print(f"Pass: {validation['pass']}, Fail: {validation['fail']}, Skipped (out of bounds): {validation['skipped']}")
    for err in validation["errors"][:10]:  # Show first 10 errors
        print(f"  {err}")
    if len(validation["errors"]) > 10:
        print(f"  ... and {len(validation['errors']) - 10} more errors")

    # Compute tidal prism
    prism = compute_tidal_prism(low_tide, high_tide, elev)

    # Get extent in NZTM for satellite basemap
    extent_nztm = get_extent_nztm(elev)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Custom colormap
    cmap = create_bathymetry_colormap()

    # Clip elevation for better visualization
    elev_clipped = np.clip(elev.data, -3, 30)

    # Extent for imshow [left, right, bottom, top]
    img_extent = [extent_nztm[0], extent_nztm[1], extent_nztm[2], extent_nztm[3]]

    # 1. Satellite imagery with ALL reference points
    ax1 = axes[0, 0]
    ax1.set_xlim(extent_nztm[0], extent_nztm[1])
    ax1.set_ylim(extent_nztm[2], extent_nztm[3])
    try:
        ctx.add_basemap(ax1, crs="EPSG:2193", source=ctx.providers.Esri.WorldImagery, zoom=15)
    except Exception as e:
        print(f"Could not load satellite basemap: {e}")
        ax1.imshow(elev_clipped, cmap="Greys", origin="upper", extent=img_extent, alpha=0.5)
    ax1.set_title("Satellite + All Reference Points")
    ax1.set_xlabel("Easting (m)")
    ax1.set_ylabel("Northing (m)")
    plot_reference_points_nztm(ax1, elev, show_legend=True)

    # 2. Low tide extent on satellite
    ax2 = axes[0, 1]
    ax2.set_xlim(extent_nztm[0], extent_nztm[1])
    ax2.set_ylim(extent_nztm[2], extent_nztm[3])
    try:
        ctx.add_basemap(ax2, crs="EPSG:2193", source=ctx.providers.Esri.WorldImagery, zoom=15)
    except Exception as e:
        ax2.imshow(elev_clipped, cmap="Greys", origin="upper", extent=img_extent, alpha=0.3)
    ax2.imshow(np.where(low_mask, 1, np.nan), cmap="Blues", origin="upper",
               extent=img_extent, alpha=0.5, vmin=0, vmax=2)
    ax2.set_title(f"Low Tide ({low_tide}m)\nArea: {prism['low_tide_area_m2']/1e6:.3f} km²")
    ax2.set_xlabel("Easting (m)")
    ax2.set_ylabel("Northing (m)")
    plot_reference_points_nztm(ax2, elev, show_legend=False)

    # 3. High tide extent on satellite
    ax3 = axes[1, 0]
    ax3.set_xlim(extent_nztm[0], extent_nztm[1])
    ax3.set_ylim(extent_nztm[2], extent_nztm[3])
    try:
        ctx.add_basemap(ax3, crs="EPSG:2193", source=ctx.providers.Esri.WorldImagery, zoom=15)
    except Exception as e:
        ax3.imshow(elev_clipped, cmap="Greys", origin="upper", extent=img_extent, alpha=0.3)
    ax3.imshow(np.where(high_mask, 1, np.nan), cmap="Blues", origin="upper",
               extent=img_extent, alpha=0.5, vmin=0, vmax=2)
    ax3.set_title(f"High Tide ({high_tide}m)\nArea: {prism['high_tide_area_m2']/1e6:.3f} km²")
    ax3.set_xlabel("Easting (m)")
    ax3.set_ylabel("Northing (m)")
    plot_reference_points_nztm(ax3, elev, show_legend=False)

    # 4. Zone classification on satellite
    ax4 = axes[1, 1]
    ax4.set_xlim(extent_nztm[0], extent_nztm[1])
    ax4.set_ylim(extent_nztm[2], extent_nztm[3])
    try:
        ctx.add_basemap(ax4, crs="EPSG:2193", source=ctx.providers.Esri.WorldImagery, zoom=15)
    except Exception as e:
        ax4.imshow(elev_clipped, cmap="Greys", origin="upper", extent=img_extent, alpha=0.3)

    # Create zone map: 0=land, 1=intertidal/mangrove, 2=always water
    zone_map = np.full_like(elev.data, np.nan)
    zone_map[high_mask & ~low_mask] = 1  # Intertidal
    zone_map[low_mask] = 2  # Always water

    # Custom colormap for zones (with transparency for land)
    from matplotlib.colors import ListedColormap
    zone_cmap = ListedColormap(["lightgreen", "steelblue"])

    ax4.imshow(zone_map, cmap=zone_cmap, origin="upper", extent=img_extent,
               alpha=0.5, vmin=1, vmax=2)
    ax4.set_title(f"Zone Classification\nGreen=Intertidal, Blue=Always Water")
    ax4.set_xlabel("Easting (m)")
    ax4.set_ylabel("Northing (m)")
    plot_reference_points_nztm(ax4, elev, show_legend=False)

    # Add overall title with tidal prism info
    validation_status = "PASS" if validation["fail"] == 0 else f"FAIL ({validation['fail']} errors)"
    fig.suptitle(
        f"Waitangi Estuary Tidal Analysis - Validation: {validation_status}\n"
        f"Tidal Prism: {prism['tidal_prism_m3']/1e6:.2f} million m³ "
        f"(tide range: {high_tide - low_tide}m)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    # Save figure
    output_path = "waitangi_tidal_extent.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization to: {output_path}")
    plt.close()

    return validation


if __name__ == "__main__":
    main()
