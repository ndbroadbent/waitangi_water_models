#!/usr/bin/env python3
"""Simulate and visualize river water flowing through Waitangi estuary.

Uses the JAX-accelerated shallow water equations solver to simulate
tidal flow with river water tracer. Green = river water, Blue = sea water.
"""

import argparse
from pathlib import Path
import subprocess

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from pyproj import Transformer
import contextily as ctx

from waitangi.data.elevation import fetch_waitangi_elevation, compute_flooded_area_at_level
from waitangi.simulation.shallow_water import ShallowWaterModel, run_simulation, SimulationState


def create_water_colormap():
    """Create colormap: blue (sea) -> cyan (mixing) -> green (river)."""
    colors = [
        (0.0, (0.1, 0.3, 0.8, 0.8)),    # Deep blue (pure sea water)
        (0.2, (0.2, 0.5, 0.9, 0.8)),    # Blue
        (0.4, (0.2, 0.7, 0.8, 0.8)),    # Cyan-blue
        (0.5, (0.2, 0.8, 0.7, 0.8)),    # Cyan (50/50 mix)
        (0.6, (0.3, 0.85, 0.5, 0.8)),   # Cyan-green
        (0.8, (0.2, 0.9, 0.3, 0.8)),    # Green
        (1.0, (0.1, 0.95, 0.1, 0.9)),   # Bright green (pure river water)
    ]
    return LinearSegmentedColormap.from_list("river_mixing", colors)


def setup_model_from_elevation(elev_data, downsample: int = 8):
    """Create shallow water model from elevation data.

    Args:
        elev_data: ElevationData from fetch_waitangi_elevation()
        downsample: Downsampling factor (8 = 8m grid from 1m DEM)

    Returns:
        ShallowWaterModel instance
    """
    # Downsample bathymetry
    bath = elev_data.data[::downsample, ::downsample]
    ny, nx = bath.shape

    # Grid spacing
    dx = abs(elev_data.transform.a) * downsample
    dy = abs(elev_data.transform.e) * downsample

    print(f"Grid: {ny} x {nx} cells, {dx:.1f}m resolution")

    # Find Haruru Falls location in grid coordinates
    haruru_lat, haruru_lon = -35.278284, 174.051297
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
    haruru_e, haruru_n = transformer.transform(haruru_lon, haruru_lat)

    river_col = int((haruru_e - elev_data.transform.c) / elev_data.transform.a) // downsample
    river_row = int((haruru_n - elev_data.transform.f) / elev_data.transform.e) // downsample

    # Adjust to find actual channel (lowest elevation nearby)
    # Search in a small window for the lowest point
    search_radius = 5
    r_min = max(0, river_row - search_radius)
    r_max = min(ny, river_row + search_radius + 1)
    c_min = max(0, river_col - search_radius)
    c_max = min(nx, river_col + search_radius + 1)

    window = bath[r_min:r_max, c_min:c_max]
    min_idx = np.unravel_index(np.argmin(window), window.shape)
    river_row = r_min + min_idx[0]
    river_col = c_min + min_idx[1]

    print(f"River source at grid ({river_row}, {river_col}), elevation: {bath[river_row, river_col]:.1f}m")

    # Clamp to valid range
    river_row = max(5, min(river_row, ny - 5))
    river_col = max(5, min(river_col, nx - 5))

    # Create mangrove mask (elevation 0.0m to 1.1m in flooded areas)
    _, flood_high = compute_flooded_area_at_level(1.1, elev_data)
    _, flood_low = compute_flooded_area_at_level(0.0, elev_data)
    mangrove_full = flood_high & ~flood_low
    mangrove_mask = mangrove_full[::downsample, ::downsample]

    print(f"Mangrove cells: {np.sum(mangrove_mask)}")

    # Create custom ocean mask - only at eastern edge where elevation is low
    # This represents the estuary mouth opening to the bay
    ocean_mask = np.zeros((ny, nx), dtype=bool)
    ocean_cols = 10  # Number of columns at boundary
    ocean_elev_threshold = 1.0  # Only cells below this elevation

    # Eastern boundary
    ocean_mask[:, -ocean_cols:] = bath[:, -ocean_cols:] < ocean_elev_threshold

    # Also southern boundary (bay opens south too)
    ocean_mask[-ocean_cols:, :] = ocean_mask[-ocean_cols:, :] | (bath[-ocean_cols:, :] < ocean_elev_threshold)

    ocean_cell_count = np.sum(ocean_mask)
    print(f"Ocean boundary cells: {ocean_cell_count}")

    # Create model
    model = ShallowWaterModel.from_arrays(
        bathymetry=bath,
        dx=dx,
        dy=dy,
        mangrove_mask=mangrove_mask,
        ocean_cols=0,  # We're using custom mask
        river_row=river_row,
        river_col=river_col,
        river_radius=4,
    )

    # Override with our custom ocean mask
    model.ocean_mask = jnp.array(ocean_mask)

    return model, elev_data


def render_frame(
    state: SimulationState,
    model: ShallowWaterModel,
    elev_data,
    frame_num: int,
    total_frames: int,
    output_dir: Path,
    tide_level: float,
    river_flow: float,
    downsample: int,
):
    """Render a single frame."""
    # Use figure size that produces even pixel dimensions at 100 dpi
    fig, ax = plt.subplots(figsize=(12.96, 9.54))  # 1296x954 -> rounds to 1296x954

    # Get extent in NZTM
    extent = [
        elev_data.bounds[0], elev_data.bounds[2],
        elev_data.bounds[1], elev_data.bounds[3]
    ]

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    # Add satellite basemap
    try:
        ctx.add_basemap(ax, crs="EPSG:2193", source=ctx.providers.Esri.WorldImagery, zoom=15)
    except Exception as e:
        print(f"Warning: Could not load basemap: {e}")

    # Get simulation data
    h = np.array(model.get_depth(state))
    tracer = np.array(state.tracer)
    speed = np.array(model.get_speed(state))

    # Upsample for display
    from scipy.ndimage import zoom
    h_full = zoom(h, downsample, order=1)
    tracer_full = zoom(tracer, downsample, order=1)

    # Trim to match original size
    h_full = h_full[:elev_data.data.shape[0], :elev_data.data.shape[1]]
    tracer_full = tracer_full[:elev_data.data.shape[0], :elev_data.data.shape[1]]

    # Wet mask
    wet = h_full > 0.05

    # Create display array
    display = np.where(wet, tracer_full, np.nan)

    # Plot
    cmap = create_water_colormap()
    img_extent = [extent[0], extent[1], extent[2], extent[3]]

    im = ax.imshow(
        display,
        cmap=cmap,
        origin='upper',
        extent=img_extent,
        vmin=0, vmax=1,
        alpha=0.85,
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label('Water Source', fontsize=10)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(['Sea', 'Mixed', 'River'])

    # Stats
    wet_area = np.sum(wet) * (abs(elev_data.transform.a) ** 2) / 1e6
    max_speed = float(jnp.max(speed))

    # Title
    time_hours = state.time / 3600
    time_mins = (state.time % 3600) / 60
    ax.set_title(
        f"Waitangi Estuary - River Water Simulation\n"
        f"Time: {int(time_hours)}h {int(time_mins)}m | "
        f"Tide: {tide_level:+.2f}m | "
        f"River: {river_flow:.1f} m³/s\n"
        f"Flooded area: {wet_area:.2f} km² | "
        f"Max speed: {max_speed:.2f} m/s | "
        f"Frame {frame_num}/{total_frames}",
        fontsize=11,
        fontweight='bold',
    )

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    plt.tight_layout()

    # Save
    frame_path = output_dir / f"frame_{frame_num:04d}.png"
    plt.savefig(frame_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()

    return frame_path


def main():
    parser = argparse.ArgumentParser(description="Simulate river water tracer through estuary")
    parser.add_argument("--duration", type=float, default=6.0, help="Duration in hours (default: 6)")
    parser.add_argument("--flow", type=float, default=1.0, help="River flow in m³/s (default: 1.0)")
    parser.add_argument("--tide-range", type=float, default=1.6, help="Tidal range in m (default: 1.6)")
    parser.add_argument("--start-phase", type=float, default=0.0, help="Tide start phase 0=low, 0.5=high (default: 0)")
    parser.add_argument("--downsample", type=int, default=8, help="Grid downsampling factor (default: 8)")
    parser.add_argument("--output-interval", type=float, default=5.0, help="Output interval in minutes (default: 5)")
    parser.add_argument("--fps", type=int, default=15, help="Video frame rate (default: 15)")
    args = parser.parse_args()

    print("=" * 60)
    print("Waitangi Estuary - Shallow Water Simulation")
    print("=" * 60)

    print("\nFetching elevation data...")
    elev = fetch_waitangi_elevation()

    print("\nSetting up model...")
    model, elev_data = setup_model_from_elevation(elev, downsample=args.downsample)

    print(f"\nSimulation parameters:")
    print(f"  Duration: {args.duration} hours")
    print(f"  River flow: {args.flow} m³/s")
    print(f"  Tide range: {args.tide_range} m")
    print(f"  Start phase: {args.start_phase} (0=low, 0.5=high)")
    print(f"  CFL timestep: {model.dt_cfl:.3f} s")

    # Run simulation
    print("\nRunning simulation...")
    tide_amplitude = args.tide_range / 2
    mean_water_level = 0.3

    results = run_simulation(
        model,
        duration_seconds=args.duration * 3600,
        tide_amplitude=tide_amplitude,
        mean_water_level=mean_water_level,
        start_phase=args.start_phase,
        river_flow=args.flow,
        output_interval=args.output_interval * 60,
    )

    print(f"\nGenerated {len(results)} output frames")

    # Create output directory
    output_dir = Path("river_tracer_frames")
    output_dir.mkdir(exist_ok=True)

    # Clear old frames
    for old_frame in output_dir.glob("frame_*.png"):
        old_frame.unlink()

    # Render frames
    print("\nRendering frames...")
    omega = 2 * jnp.pi / (12.42 * 3600)
    phase0 = args.start_phase * 2 * jnp.pi

    for i, state in enumerate(results):
        tide_level = mean_water_level + tide_amplitude * float(jnp.sin(omega * state.time + phase0))
        print(f"  Frame {i+1}/{len(results)} - t={state.time/3600:.2f}h")
        render_frame(
            state, model, elev_data,
            i + 1, len(results),
            output_dir,
            tide_level, args.flow,
            args.downsample,
        )

    # Create video
    video_path = "river_tracer.mp4"
    print(f"\nCreating video: {video_path}")

    try:
        result = subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(args.fps),
            "-i", str(output_dir / "frame_%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            video_path,
        ], check=True, capture_output=True, text=True)
        print(f"Video saved to: {video_path}")
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error: {e.stderr}")
    except FileNotFoundError:
        print("ffmpeg not found - frames saved but video not created")

    print(f"\nDone! Frames saved to: {output_dir}/")


if __name__ == "__main__":
    main()
