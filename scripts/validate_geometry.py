#!/usr/bin/env python3
"""Validate OSM geometry against known reference points.

This script plots the OSM data with reference points overlaid
to visually verify that water/land classification is correct.
"""

import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer

from waitangi.core.config import WaitangiLocation
from waitangi.data.osm import fetch_waitangi_geometry_sync, get_cached_geometry, save_geometry_cache
from waitangi.data.reference_points import (
    ALL_WATER_POINTS,
    LAND_POINTS,
    LANDMARKS,
    WATER_POINTS_EAST,
    WATER_POINTS_WEST,
    get_bounding_box,
)


def main():
    # Load OSM geometry
    print("Loading OSM geometry...")
    osm_data = get_cached_geometry()
    if osm_data is None:
        print("Fetching from Overpass API...")
        osm_data = fetch_waitangi_geometry_sync()
        save_geometry_cache(osm_data)

    # Set up coordinate transformer
    transformer = Transformer.from_crs(
        WaitangiLocation.CRS_WGS84, WaitangiLocation.CRS_NZTM, always_xy=True
    )

    def transform_coords(coords):
        """Transform list of (lat, lon) to (x, y) in NZTM."""
        result = []
        for lat, lon in coords:
            x, y = transformer.transform(lon, lat)
            result.append((x, y))
        return result

    def transform_point(lat, lon):
        """Transform single point."""
        return transformer.transform(lon, lat)

    # Transform all geometry
    river_segments_nztm = [
        transform_coords(seg) for seg in osm_data.get("river_segments", [])
    ]
    coastline_segments_nztm = [
        transform_coords(seg) for seg in osm_data.get("coastline_segments", [])
    ]
    water_polygons_nztm = [
        transform_coords(seg) for seg in osm_data.get("water_polygons", [])
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Draw land background
    ax.set_facecolor('#3d5c3d')

    # Draw water polygons
    for poly in water_polygons_nztm:
        if poly:
            xs, ys = zip(*poly)
            ax.fill(xs, ys, color='#1a5f8a', alpha=0.9, zorder=1, label='Water polygon')
            ax.plot(xs, ys, color='#0d3d5c', linewidth=0.5, zorder=1)

    # Draw coastline polygons (land)
    for i, seg in enumerate(coastline_segments_nztm):
        if seg:
            xs, ys = zip(*seg)
            label = 'Coastline (land)' if i == 0 else None
            ax.fill(xs, ys, color='#c4a574', alpha=0.95, zorder=2, label=label)
            ax.plot(xs, ys, color='#8b7355', linewidth=1, zorder=2)

    # Draw river channel
    for i, seg in enumerate(river_segments_nztm):
        if seg:
            xs, ys = zip(*seg)
            label = 'River channel' if i == 0 else None
            ax.plot(xs, ys, color='#1a5f8a', linewidth=10, alpha=0.9,
                   solid_capstyle='round', zorder=3, label=label)
            ax.plot(xs, ys, color='#2d7ab8', linewidth=5, alpha=0.7,
                   solid_capstyle='round', zorder=3)

    # Plot reference points
    # Water points west (should be blue/in water)
    for i, pt in enumerate(WATER_POINTS_WEST):
        x, y = transform_point(pt.lat, pt.lon)
        label = 'Water ref (west)' if i == 0 else None
        ax.plot(x, y, 'o', color='cyan', markersize=8, markeredgecolor='white',
               markeredgewidth=1, zorder=10, label=label)

    # Water points east (should be blue/in water)
    for i, pt in enumerate(WATER_POINTS_EAST):
        x, y = transform_point(pt.lat, pt.lon)
        label = 'Water ref (east)' if i == 0 else None
        ax.plot(x, y, 's', color='deepskyblue', markersize=8, markeredgecolor='white',
               markeredgewidth=1, zorder=10, label=label)

    # Land points (should be on brown/green land)
    for i, pt in enumerate(LAND_POINTS):
        x, y = transform_point(pt.lat, pt.lon)
        label = 'Land ref' if i == 0 else None
        ax.plot(x, y, '^', color='red', markersize=8, markeredgecolor='white',
               markeredgewidth=1, zorder=10, label=label)

    # Landmarks
    for pt in LANDMARKS:
        x, y = transform_point(pt.lat, pt.lon)
        ax.plot(x, y, '*', color='yellow', markersize=15, markeredgecolor='black',
               markeredgewidth=1, zorder=11)
        ax.annotate(pt.name, (x, y), textcoords="offset points",
                   xytext=(8, 5), fontsize=8, color='white', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

    # Get bounds from reference points
    bbox = get_bounding_box()
    min_lat, min_lon, max_lat, max_lon = bbox

    # Transform bounds
    x_min, y_min = transform_point(min_lat, min_lon)
    x_max, y_max = transform_point(max_lat, max_lon)

    # Add margin
    margin = 500  # meters
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)

    ax.set_aspect('equal')
    ax.set_xlabel('Easting (m NZTM)', fontsize=10)
    ax.set_ylabel('Northing (m NZTM)', fontsize=10)
    ax.set_title('Waitangi River Geometry Validation\n'
                 'Cyan/Blue = known water points, Red = known land points, Yellow = landmarks',
                 fontsize=12)

    # Legend
    ax.legend(loc='upper right', fontsize=8)

    # Summary stats
    print(f"\nGeometry summary:")
    print(f"  River segments: {len(river_segments_nztm)}")
    print(f"  Coastline segments: {len(coastline_segments_nztm)}")
    print(f"  Water polygons: {len(water_polygons_nztm)}")
    print(f"\nReference points:")
    print(f"  Water points (west): {len(WATER_POINTS_WEST)}")
    print(f"  Water points (east): {len(WATER_POINTS_EAST)}")
    print(f"  Land points: {len(LAND_POINTS)}")
    print(f"  Landmarks: {len(LANDMARKS)}")

    # Check if reference points appear correct
    print(f"\nBounding box (lat/lon): {bbox}")

    plt.tight_layout()
    plt.savefig('geometry_validation.png', dpi=150, facecolor='#1a1a2e')
    print(f"\nSaved validation plot to: geometry_validation.png")
    plt.show()


if __name__ == "__main__":
    main()
