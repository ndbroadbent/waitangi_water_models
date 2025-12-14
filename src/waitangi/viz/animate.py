"""Real-time animation of kayak simulation with ffmpeg streaming."""

import subprocess
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer

from waitangi.core.config import WaitangiLocation


def render_frame_to_pipe(fig, pipe, dpi: int = 100):
    """Render matplotlib figure directly to ffmpeg pipe."""
    buf = BytesIO()
    fig.savefig(buf, format='raw', dpi=dpi)
    buf.seek(0)
    pipe.write(buf.getvalue())
    buf.close()


def animate_kayak_simulation(
    output_path: Path,
    duration_hours: float = 2.0,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    paddle_speed_kmh: float = 0.0,
    direction: str = "upstream",
    start_chainage: float = 500.0,
):
    """Create animated video of kayak simulation.

    Streams frames directly to ffmpeg - no intermediate files.
    Uses real OpenStreetMap geometry for the Waitangi River.
    """
    from waitangi.simulation.runner import SimulationRunner
    from waitangi.simulation.kayak import PaddlingProfile
    from waitangi.core.constants import KMH_TO_MS
    from waitangi.data.osm import fetch_waitangi_geometry_sync, get_cached_geometry
    import asyncio

    print("Initializing simulation...")

    # Fetch real geometry from OSM
    print("Loading Waitangi River geometry from OpenStreetMap...")
    osm_data = get_cached_geometry()
    if osm_data is None:
        osm_data = fetch_waitangi_geometry_sync()
        from waitangi.data.osm import save_geometry_cache
        save_geometry_cache(osm_data)

    # Initialize runner
    async def init():
        runner = SimulationRunner()
        await runner.initialize(use_synthetic_data=True)
        return runner

    runner = asyncio.run(init())
    mesh = runner.mesh

    # Coordinate transformer (WGS84 to NZTM for plotting)
    transformer = Transformer.from_crs(
        WaitangiLocation.CRS_WGS84, WaitangiLocation.CRS_NZTM, always_xy=True
    )

    # Transform OSM geometry to NZTM
    def transform_coords(coords):
        """Transform list of (lat, lon) to (x, y) in NZTM."""
        result = []
        for lat, lon in coords:
            x, y = transformer.transform(lon, lat)
            result.append((x, y))
        return result

    river_segments_nztm = [
        transform_coords(seg) for seg in osm_data.get("river_segments", [])
    ]
    coastline_segments_nztm = [
        transform_coords(seg) for seg in osm_data.get("coastline_segments", [])
    ]
    water_polygons_nztm = [
        transform_coords(seg) for seg in osm_data.get("water_polygons", [])
    ]

    # Key landmarks
    landmarks = osm_data.get("landmarks", {})
    bridge_pos = landmarks.get("bridge")
    bridge_paihia_pos = landmarks.get("bridge_paihia")
    bridge_waitangi_pos = landmarks.get("bridge_waitangi")
    slipway_pos = landmarks.get("slipway")
    mouth_pos = landmarks.get("mouth")

    # Bridge center point
    if bridge_pos:
        bridge_x, bridge_y = transformer.transform(bridge_pos[1], bridge_pos[0])
    else:
        bridge_x, bridge_y = None, None

    # Bridge endpoints for drawing the bridge line
    if bridge_paihia_pos and bridge_waitangi_pos:
        bridge_paihia_x, bridge_paihia_y = transformer.transform(bridge_paihia_pos[1], bridge_paihia_pos[0])
        bridge_waitangi_x, bridge_waitangi_y = transformer.transform(bridge_waitangi_pos[1], bridge_waitangi_pos[0])
    else:
        bridge_paihia_x, bridge_paihia_y = None, None
        bridge_waitangi_x, bridge_waitangi_y = None, None

    if slipway_pos:
        slipway_x, slipway_y = transformer.transform(slipway_pos[1], slipway_pos[0])
    else:
        slipway_x, slipway_y = None, None

    if mouth_pos:
        mouth_x, mouth_y = transformer.transform(mouth_pos[1], mouth_pos[0])
    else:
        mouth_x, mouth_y = float(mesh.river_centerline_x[0]), float(mesh.river_centerline_y[0])

    # Get start position
    start_x, start_y = mesh.chainage_to_point(start_chainage)

    # Create paddling profile
    profile = None
    if paddle_speed_kmh > 0:
        profile = PaddlingProfile(
            mode="constant",
            base_speed=paddle_speed_kmh * KMH_TO_MS,
            heading_mode=direction,
        )

    # Simulation parameters
    sim_dt = 1.0
    duration_seconds = duration_hours * 3600
    n_sim_steps = int(duration_seconds / sim_dt)

    # Video parameters
    video_duration = min(60.0, duration_hours * 60)
    n_frames = int(video_duration * fps)
    sim_steps_per_frame = max(1, n_sim_steps // n_frames)

    print(f"Simulation: {n_sim_steps} steps over {duration_hours}h")
    print(f"Video: {n_frames} frames at {fps}fps = {video_duration:.1f}s")
    print(f"Time compression: {duration_hours * 3600 / video_duration:.0f}x")

    # Set up figure
    dpi = 100
    fig_width = width / dpi
    fig_height = height / dpi

    fig, (ax_map, ax_info) = plt.subplots(
        1, 2,
        figsize=(fig_width, fig_height),
        gridspec_kw={'width_ratios': [3, 1]},
    )

    # Compute bounds from all geometry
    all_x = []
    all_y = []
    for seg in river_segments_nztm:
        for x, y in seg:
            all_x.append(x)
            all_y.append(y)
    for seg in coastline_segments_nztm:
        for x, y in seg:
            all_x.append(x)
            all_y.append(y)

    if not all_x:
        # Fallback to mesh bounds
        all_x = list(np.asarray(mesh.river_centerline_x))
        all_y = list(np.asarray(mesh.river_centerline_y))

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # Focus on the estuary/mouth area for better visualization
    # Limit to ~5km around the mouth
    focus_range = 5000  # 5km
    x_center = mouth_x if mouth_x else (x_min + x_max) / 2
    y_center = mouth_y if mouth_y else (y_min + y_max) / 2

    x_min = max(x_min, x_center - focus_range)
    x_max = min(x_max, x_center + focus_range)
    y_min = max(y_min, y_center - focus_range)
    y_max = min(y_max, y_center + focus_range)

    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1

    # Start ffmpeg process
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'rgba',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        str(output_path),
    ]

    print(f"Starting ffmpeg...")

    pipe = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Initialize kayak
    from waitangi.simulation.kayak import KayakSimulator, KayakState

    start_time = datetime.now()
    initial_state = KayakState(x=start_x, y=start_y, timestamp=start_time)
    simulator = KayakSimulator(
        initial_state=initial_state,
        paddling_profile=profile or PaddlingProfile.no_paddle(),
    )

    trajectory_x = [start_x]
    trajectory_y = [start_y]

    def get_velocities(x, y, t):
        water_uv = runner.velocity_field.get_velocity_at_point(x, y, t)
        wind_uv = runner.wind_model.get_kayak_drift(t)
        chainage = mesh.point_to_chainage(x, y)
        river_dir = runner.velocity_field._get_river_direction(chainage)
        return water_uv, wind_uv, river_dir

    print(f"Rendering {n_frames} frames...")

    try:
        for frame_idx in range(n_frames):
            # Run simulation steps
            for _ in range(sim_steps_per_frame):
                simulator.step_rk2(sim_dt, get_velocities)

            trajectory_x.append(simulator.state.x)
            trajectory_y.append(simulator.state.y)

            # Clear axes
            ax_map.clear()
            ax_info.clear()

            # Draw land background (green/brown terrain)
            ax_map.set_facecolor('#3d5c3d')  # Dark green land

            # Draw water polygons (estuaries, bays) in blue
            for poly in water_polygons_nztm:
                if poly:
                    xs, ys = zip(*poly)
                    ax_map.fill(xs, ys, color='#1a5f8a', alpha=0.9, zorder=1)
                    ax_map.plot(xs, ys, color='#0d3d5c', linewidth=0.5, zorder=1)

            # Draw coastline polygons - these are land/islands on top of water
            for seg in coastline_segments_nztm:
                if seg:
                    xs, ys = zip(*seg)
                    ax_map.fill(xs, ys, color='#c4a574', alpha=0.95, zorder=2)  # Sandy land
                    ax_map.plot(xs, ys, color='#8b7355', linewidth=1, zorder=2)

            # Draw river channel (blue water)
            for seg in river_segments_nztm:
                if seg:
                    xs, ys = zip(*seg)
                    # Wide blue channel
                    ax_map.plot(xs, ys, color='#1a5f8a', linewidth=12, alpha=0.9, solid_capstyle='round', zorder=3)
                    # Lighter center for depth effect
                    ax_map.plot(xs, ys, color='#2d7ab8', linewidth=6, alpha=0.7, solid_capstyle='round', zorder=3)

            # Draw mesh centerline (thin line)
            cx = np.asarray(mesh.river_centerline_x)
            cy = np.asarray(mesh.river_centerline_y)
            ax_map.plot(cx, cy, 'c--', linewidth=0.5, alpha=0.3, label='Sim centerline')

            # Draw trajectory
            if len(trajectory_x) > 1:
                n_pts = len(trajectory_x)
                trail_len = min(n_pts, 300)
                for i in range(1, trail_len):
                    idx = n_pts - trail_len + i
                    if idx > 0:
                        alpha = 0.1 + 0.6 * (i / trail_len)
                        ax_map.plot(
                            [trajectory_x[idx-1], trajectory_x[idx]],
                            [trajectory_y[idx-1], trajectory_y[idx]],
                            color='orange',
                            linewidth=2.5,
                            alpha=alpha,
                        )

            # Draw kayak
            kayak_x = simulator.state.x
            kayak_y = simulator.state.y

            # Kayak marker
            ax_map.plot(kayak_x, kayak_y, 'o', color='red', markersize=10, zorder=10)
            ax_map.plot(kayak_x, kayak_y, 'o', color='yellow', markersize=6, zorder=11)

            # Velocity arrow
            arrow_scale = 30
            vx, vy = simulator.state.vx, simulator.state.vy
            if abs(vx) > 0.01 or abs(vy) > 0.01:
                ax_map.annotate(
                    '',
                    xy=(kayak_x + vx * arrow_scale, kayak_y + vy * arrow_scale),
                    xytext=(kayak_x, kayak_y),
                    arrowprops=dict(arrowstyle='->', color='yellow', lw=2),
                    zorder=12,
                )

            # Mark landmarks
            # Draw bridge as a line spanning the tidal channel
            if bridge_paihia_x is not None and bridge_waitangi_x is not None:
                # Bridge structure (brown wooden bridge)
                ax_map.plot(
                    [bridge_paihia_x, bridge_waitangi_x],
                    [bridge_paihia_y, bridge_waitangi_y],
                    color='#8B4513', linewidth=6, solid_capstyle='butt', zorder=8
                )
                ax_map.plot(
                    [bridge_paihia_x, bridge_waitangi_x],
                    [bridge_paihia_y, bridge_waitangi_y],
                    color='#A0522D', linewidth=3, solid_capstyle='butt', zorder=8
                )
                # Label at center
                if bridge_x and x_min < bridge_x < x_max:
                    ax_map.annotate('Waitangi Bridge', (bridge_x, bridge_y), textcoords="offset points",
                                   xytext=(5, 8), fontsize=8, color='white', fontweight='bold')

            if slipway_x and x_min < slipway_x < x_max:
                ax_map.plot(slipway_x, slipway_y, '^', color='#32CD32', markersize=10, zorder=9)
                ax_map.plot(slipway_x, slipway_y, 'v', color='#228B22', markersize=6, zorder=9)
                ax_map.annotate('Boat Ramp', (slipway_x, slipway_y), textcoords="offset points",
                               xytext=(8, 0), fontsize=8, color='white', fontweight='bold')

            ax_map.plot(start_x, start_y, 'ws', markersize=8)
            ax_map.annotate('Start', (start_x, start_y), textcoords="offset points",
                           xytext=(5, -10), fontsize=8, color='white')

            # Set limits
            ax_map.set_xlim(x_min - x_margin, x_max + x_margin)
            ax_map.set_ylim(y_min - y_margin, y_max + y_margin)
            ax_map.set_aspect('equal')
            ax_map.set_xlabel('Easting (m)', color='white')
            ax_map.set_ylabel('Northing (m)', color='white')
            ax_map.tick_params(colors='white')

            sim_time = start_time + timedelta(seconds=frame_idx * sim_steps_per_frame * sim_dt)
            ax_map.set_title(f'Waitangi River - {sim_time.strftime("%H:%M:%S")}', color='white', fontsize=14)
            ax_map.set_facecolor('#1a3a5c')

            # Info panel
            ax_info.axis('off')
            ax_info.set_facecolor('#1a1a2e')

            chainage = mesh.point_to_chainage(kayak_x, kayak_y)
            tide_height = runner.tide_model.get_height(sim_time)
            tide_vel = runner.tide_model.get_velocity(sim_time)
            tide_phase = runner.tide_model.get_phase(sim_time)
            river_vel = runner.river_model.get_velocity_at_mouth(sim_time)
            wind_speed, wind_dir = runner.wind_model.get_wind(sim_time)

            speed_ms = simulator.state.speed_over_ground
            speed_kmh = speed_ms * 3.6

            info_text = f"""
KAYAK
───────────────
Position: {chainage:.0f}m from mouth
Speed: {speed_kmh:.1f} km/h
Paddling: {paddle_speed_kmh:.1f} km/h {direction if paddle_speed_kmh > 0 else '(drift)'}

TIDE
───────────────
Height: {tide_height:.2f} m
Current: {tide_vel:+.2f} m/s
Phase: {tide_phase}

RIVER
───────────────
Flow: {river_vel:.2f} m/s

WIND
───────────────
{wind_speed:.1f} m/s @ {wind_dir:.0f}°

TIME
───────────────
Elapsed: {frame_idx * sim_steps_per_frame * sim_dt / 60:.1f} min
"""

            ax_info.text(
                0.05, 0.95, info_text,
                transform=ax_info.transAxes,
                fontsize=10,
                fontfamily='monospace',
                verticalalignment='top',
                color='white',
            )

            fig.patch.set_facecolor('#1a1a2e')
            fig.tight_layout()

            render_frame_to_pipe(fig, pipe.stdin, dpi=dpi)

            if (frame_idx + 1) % 30 == 0 or frame_idx == n_frames - 1:
                pct = 100 * (frame_idx + 1) / n_frames
                print(f"  Frame {frame_idx + 1}/{n_frames} ({pct:.0f}%)")

        print("Finalizing video...")

    finally:
        pipe.stdin.close()
        pipe.wait()
        plt.close(fig)

    print(f"Video saved to: {output_path}")
    return output_path
