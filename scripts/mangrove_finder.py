#!/usr/bin/env python3
"""Generate video frames from low tide to high tide to find mangrove elevation threshold.

Mangroves establish in a specific elevation band - they need regular tidal flooding
but cannot survive permanent submersion. This script creates frames at different
water levels so you can identify the exact threshold where sandbars disappear
and only mangroves remain in the intertidal zone.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from pyproj import Transformer
import contextily as ctx
import subprocess

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


def latlon_to_nztm(lat: float, lon: float) -> tuple[float, float]:
    """Convert lat/lon to NZTM coordinates."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    return easting, northing


def plot_reference_points_nztm(ax, elev_data):
    """Plot all reference points on an axis (NZTM coordinates)."""
    point_groups = [
        (WATER_POINTS_WEST + WATER_POINTS_EAST, "cyan", "o", "Water", 20),
        (MANGROVE_POINTS, "lime", "^", "Mangrove", 25),
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


def get_extent_nztm(elev_data) -> list[float]:
    """Get the extent of elevation data in NZTM coordinates."""
    return [elev_data.bounds[0], elev_data.bounds[2],
            elev_data.bounds[1], elev_data.bounds[3]]


def generate_frame(
    elev_data,
    water_level: float,
    low_tide: float,
    high_tide: float,
    frame_num: int,
    total_frames: int,
    output_dir: Path,
) -> Path:
    """Generate a single frame showing intertidal zone at given water level."""
    # Compute flood masks
    _, current_mask = compute_flooded_area_at_level(water_level, elev_data)
    _, high_mask = compute_flooded_area_at_level(high_tide, elev_data)

    # Calculate areas
    pixel_area_m2 = abs(elev_data.transform.a * elev_data.transform.e)
    flooded_area_km2 = np.sum(current_mask) * pixel_area_m2 / 1e6

    # Intertidal zone: flooded at high tide but not at current level
    intertidal_mask = high_mask & ~current_mask
    intertidal_area_km2 = np.sum(intertidal_mask) * pixel_area_m2 / 1e6

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))

    extent_nztm = get_extent_nztm(elev_data)
    img_extent = [extent_nztm[0], extent_nztm[1], extent_nztm[2], extent_nztm[3]]

    ax.set_xlim(extent_nztm[0], extent_nztm[1])
    ax.set_ylim(extent_nztm[2], extent_nztm[3])

    # Add satellite basemap
    try:
        ctx.add_basemap(ax, crs="EPSG:2193", source=ctx.providers.Esri.WorldImagery, zoom=15)
    except Exception as e:
        print(f"Could not load basemap: {e}")

    # Overlay current water (blue)
    ax.imshow(np.where(current_mask, 1, np.nan), cmap="Blues", origin="upper",
              extent=img_extent, alpha=0.6, vmin=0, vmax=2)

    # Overlay intertidal zone (green) - this is what will become mangrove candidates
    ax.imshow(np.where(intertidal_mask, 1, np.nan), cmap="Greens", origin="upper",
              extent=img_extent, alpha=0.5, vmin=0, vmax=2)

    # Plot reference points
    plot_reference_points_nztm(ax, elev_data)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", alpha=0.6, label="Currently flooded"),
        Patch(facecolor="green", alpha=0.5, label="Exposed (intertidal)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.9)

    # Title with water level info
    ax.set_title(
        f"Water Level: {water_level:+.2f}m  |  Frame {frame_num}/{total_frames}\n"
        f"Flooded: {flooded_area_km2:.3f} km²  |  "
        f"Exposed intertidal: {intertidal_area_km2:.3f} km²\n"
        f"(Low tide: {low_tide}m → High tide: {high_tide}m)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    plt.tight_layout()

    # Save frame
    frame_path = output_dir / f"frame_{frame_num:04d}.png"
    plt.savefig(frame_path, dpi=120, bbox_inches="tight")
    plt.close()

    return frame_path


def main():
    print("Fetching elevation data...")
    elev = fetch_waitangi_elevation()

    # Tide range
    low_tide = -0.5
    high_tide = 1.1

    # Generate frames at 0.05m intervals
    step = 0.05
    water_levels = np.arange(low_tide, high_tide + step, step)
    total_frames = len(water_levels)

    print(f"Generating {total_frames} frames from {low_tide}m to {high_tide}m...")

    # Create output directory
    output_dir = Path("mangrove_finder_frames")
    output_dir.mkdir(exist_ok=True)

    # Generate frames
    for i, water_level in enumerate(water_levels, 1):
        print(f"  Frame {i}/{total_frames}: water level = {water_level:+.2f}m")
        generate_frame(elev, water_level, low_tide, high_tide, i, total_frames, output_dir)

    print(f"\nFrames saved to: {output_dir}/")

    # Create video using ffmpeg
    video_path = "mangrove_finder.mp4"
    print(f"\nCreating video: {video_path}")

    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", "4",  # 4 fps - slow enough to see each frame
            "-i", str(output_dir / "frame_%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            video_path,
        ], check=True, capture_output=True)
        print(f"Video saved to: {video_path}")
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error: {e.stderr.decode()}")
    except FileNotFoundError:
        print("ffmpeg not found - video not created, but frames are available")

    # Print frame index for reference
    print("\n=== Frame Reference ===")
    for i, water_level in enumerate(water_levels, 1):
        print(f"Frame {i:3d}: {water_level:+.2f}m")


if __name__ == "__main__":
    main()
