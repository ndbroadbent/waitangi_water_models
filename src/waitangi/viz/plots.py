"""Plotting functions for simulation visualization."""

from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from waitangi.simulation.runner import SimulationResult


def plot_trajectory(
    result: SimulationResult,
    mesh=None,
    show_speed: bool = True,
    save_path: Path | None = None,
) -> Figure:
    """Plot kayak trajectory on river map.

    Args:
        result: Simulation result with trajectory data.
        mesh: Optional mesh for river outline.
        show_speed: Color trajectory by speed.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.array(result.trajectory_x)
    y = np.array(result.trajectory_y)
    speed = np.array(result.trajectory_speed)

    if show_speed:
        # Color by speed
        points = ax.scatter(
            x, y,
            c=speed,
            cmap="viridis",
            s=2,
            alpha=0.8,
        )
        cbar = plt.colorbar(points, ax=ax)
        cbar.set_label("Speed (m/s)")
    else:
        ax.plot(x, y, "b-", linewidth=1, alpha=0.8)

    # Mark start and end
    ax.plot(x[0], y[0], "go", markersize=10, label="Start")
    ax.plot(x[-1], y[-1], "ro", markersize=10, label="End")

    # Plot river centerline if mesh provided
    if mesh is not None:
        ax.plot(
            np.asarray(mesh.river_centerline_x),
            np.asarray(mesh.river_centerline_y),
            "k--",
            linewidth=0.5,
            alpha=0.5,
            label="Centerline",
        )

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(
        f"Kayak Trajectory\n"
        f"{result.start_time.strftime('%Y-%m-%d %H:%M')} - "
        f"{result.end_time.strftime('%H:%M')}"
    )
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Add statistics annotation
    stats_text = (
        f"Distance: {result.total_distance_m:.0f} m\n"
        f"Displacement: {result.net_displacement_m:.0f} m\n"
        f"Mean speed: {result.mean_speed_ms:.2f} m/s\n"
        f"Max speed: {result.max_speed_ms:.2f} m/s"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_velocity_profile(
    profile: dict,
    save_path: Path | None = None,
) -> Figure:
    """Plot velocity components along river.

    Args:
        profile: Dictionary from VelocityField.get_velocity_profile().
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    chainage_km = np.asarray(profile["chainage_m"]) / 1000

    ax.plot(
        chainage_km,
        np.asarray(profile["v_tide_ms"]),
        "b-",
        linewidth=2,
        label="Tide (+ = flood)",
    )
    ax.plot(
        chainage_km,
        -np.asarray(profile["v_river_ms"]),
        "r-",
        linewidth=2,
        label="River (- = downstream)",
    )
    ax.plot(
        chainage_km,
        np.asarray(profile["v_net_ms"]),
        "k-",
        linewidth=2,
        label="Net velocity",
    )

    # Mark cancellation zone
    cancel_km = profile["cancellation_m"] / 1000
    ax.axvline(cancel_km, color="gray", linestyle="--", alpha=0.7)
    ax.text(
        cancel_km, ax.get_ylim()[1] * 0.9,
        f" Stagnation\n {cancel_km:.1f} km",
        fontsize=9,
    )

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Distance from mouth (km)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Velocity Profile Along River")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_conditions_dashboard(
    tide_model,
    river_model,
    wind_model,
    hours_ahead: int = 12,
    save_path: Path | None = None,
) -> Figure:
    """Create a dashboard of current and forecast conditions.

    Args:
        tide_model: TideModel instance.
        river_model: RiverDischargeModel instance.
        wind_model: WindModel instance.
        hours_ahead: Hours to forecast.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    now = datetime.now()
    times = [now + timedelta(hours=h) for h in np.linspace(0, hours_ahead, 100)]

    # Tide panel
    ax = axes[0]
    heights = [tide_model.get_height(t) for t in times]
    velocities = [tide_model.get_velocity(t) for t in times]

    ax.plot(times, heights, "b-", linewidth=2, label="Height (m)")
    ax.axhline(tide_model.harmonics.z0, color="b", linestyle="--", alpha=0.5)

    ax2 = ax.twinx()
    ax2.plot(times, velocities, "r-", linewidth=1, alpha=0.7, label="Velocity (m/s)")
    ax2.axhline(0, color="r", linestyle="--", alpha=0.3)

    ax.set_ylabel("Tide height (m)", color="b")
    ax2.set_ylabel("Tidal velocity (m/s)", color="r")
    ax.set_title("Tides")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # River flow panel
    ax = axes[1]
    flows = [river_model.get_discharge(t) for t in times]
    velocities = [river_model.get_velocity_at_mouth(t) for t in times]

    ax.plot(times, flows, "g-", linewidth=2, label="Discharge (m³/s)")

    ax2 = ax.twinx()
    ax2.plot(times, velocities, "m-", linewidth=1, alpha=0.7, label="Velocity (m/s)")

    ax.set_ylabel("Discharge (m³/s)", color="g")
    ax2.set_ylabel("Velocity at mouth (m/s)", color="m")
    ax.set_title("River Flow")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Wind panel
    ax = axes[2]
    speeds = []
    directions = []
    for t in times:
        s, d = wind_model.get_wind(t)
        speeds.append(s)
        directions.append(d)

    ax.plot(times, speeds, "c-", linewidth=2, label="Wind speed (m/s)")

    ax2 = ax.twinx()
    ax2.scatter(times, directions, c="orange", s=5, alpha=0.5, label="Direction (°)")
    ax2.set_ylim(0, 360)

    ax.set_ylabel("Wind speed (m/s)", color="c")
    ax2.set_ylabel("Wind direction (°)", color="orange")
    ax.set_xlabel("Time")
    ax.set_title("Wind")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Format x-axis
    import matplotlib.dates as mdates
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_particle_evolution(
    history: list,
    mesh=None,
    n_frames: int = 10,
    save_path: Path | None = None,
) -> Figure:
    """Plot evolution of particle cloud over time.

    Args:
        history: List of ParticleState objects.
        mesh: Optional mesh for river outline.
        n_frames: Number of time snapshots to show.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    n_states = len(history)
    indices = np.linspace(0, n_states - 1, n_frames, dtype=int)

    fig, axes = plt.subplots(2, (n_frames + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        ax = axes[i]
        state = history[idx]

        # Plot particles
        active_mask = np.asarray(state.active) > 0.5
        x = np.asarray(state.x)[active_mask]
        y = np.asarray(state.y)[active_mask]
        speed = np.sqrt(
            np.asarray(state.vx)[active_mask]**2 +
            np.asarray(state.vy)[active_mask]**2
        )

        scatter = ax.scatter(x, y, c=speed, cmap="viridis", s=1, alpha=0.5)

        # Plot river centerline
        if mesh is not None:
            ax.plot(
                np.asarray(mesh.river_centerline_x),
                np.asarray(mesh.river_centerline_y),
                "k--",
                linewidth=0.5,
                alpha=0.3,
            )

        ax.set_aspect("equal")
        ax.set_title(f"t = {state.timestamp.strftime('%H:%M')}")
        ax.tick_params(labelsize=8)

    # Remove unused axes
    for i in range(n_frames, len(axes)):
        axes[i].set_visible(False)

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=axes, orientation="horizontal", pad=0.05, shrink=0.5)
    cbar.set_label("Speed (m/s)")

    fig.suptitle("Particle Cloud Evolution", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
