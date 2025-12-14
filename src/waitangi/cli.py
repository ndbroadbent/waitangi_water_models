"""Command-line interface for the Waitangi River Kayak Simulator."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="waitangi",
    help="Waitangi River Kayak Tidal Simulator",
    add_completion=False,
)
console = Console()


@app.command()
def conditions(
    hours_ahead: Annotated[int, typer.Option(help="Hours to forecast")] = 12,
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON")] = False,
):
    """Show current environmental conditions."""
    from waitangi.simulation.runner import SimulationRunner

    async def _run():
        runner = SimulationRunner()
        await runner.initialize(use_synthetic_data=True)
        return runner.get_current_conditions()

    conditions = asyncio.run(_run())

    if json_output:
        import json
        console.print(json.dumps(conditions, indent=2, default=str))
        return

    # Rich output
    console.print(Panel.fit(
        f"[bold]Environmental Conditions[/bold]\n"
        f"Time: {conditions['timestamp'].strftime('%Y-%m-%d %H:%M')}",
    ))

    # Tide table
    if "tide" in conditions:
        tide = conditions["tide"]
        table = Table(title="Tide")
        table.add_column("Property")
        table.add_column("Value")
        table.add_row("Height", f"{tide['height_m']:.2f} m")
        table.add_row("Velocity", f"{tide['velocity_ms']:.3f} m/s")
        table.add_row("Phase", tide["phase"])
        console.print(table)

    # River table
    if "river" in conditions:
        river = conditions["river"]
        table = Table(title="River")
        table.add_column("Property")
        table.add_column("Value")
        table.add_row("Discharge", f"{river['discharge_m3s']:.1f} m³/s")
        table.add_row("Velocity at mouth", f"{river['velocity_at_mouth_ms']:.2f} m/s")
        console.print(table)

    # Wind table
    if "wind" in conditions:
        wind = conditions["wind"]
        table = Table(title="Wind")
        table.add_column("Property")
        table.add_column("Value")
        table.add_row("Speed", f"{wind['speed_ms']:.1f} m/s")
        table.add_row("Direction", f"{wind['direction_deg']:.0f}°")
        table.add_row("Description", wind["description"])
        console.print(table)

    # Cancellation zone
    if "cancellation_zone_m" in conditions:
        console.print(
            f"\n[yellow]Stagnation zone:[/yellow] "
            f"{conditions['cancellation_zone_m']:.0f} m from mouth"
        )


@app.command()
def simulate(
    duration: Annotated[float, typer.Option(help="Duration in hours")] = 4.0,
    paddle_speed: Annotated[float, typer.Option(help="Paddling speed km/h")] = 0.0,
    direction: Annotated[str, typer.Option(help="Direction: upstream/downstream")] = "upstream",
    start_chainage: Annotated[float, typer.Option(help="Start distance from mouth (m)")] = 500.0,
    output: Annotated[Optional[Path], typer.Option(help="Output JSON file")] = None,
    plot: Annotated[bool, typer.Option(help="Show trajectory plot")] = False,
):
    """Run a kayak simulation."""
    from waitangi.simulation.kayak import PaddlingProfile
    from waitangi.simulation.runner import SimulationRunner
    from waitangi.core.constants import KMH_TO_MS

    async def _run():
        runner = SimulationRunner()
        await runner.initialize(use_synthetic_data=True)

        # Get start position from chainage
        start_x, start_y = runner.mesh.chainage_to_point(start_chainage)

        # Create paddling profile
        profile = None
        if paddle_speed > 0:
            if direction == "upstream":
                profile = PaddlingProfile(
                    mode="constant",
                    base_speed=paddle_speed * KMH_TO_MS,
                    heading_mode="upstream",
                )
            else:
                profile = PaddlingProfile(
                    mode="constant",
                    base_speed=paddle_speed * KMH_TO_MS,
                    heading_mode="downstream",
                )

        # Run simulation
        result = runner.run_single_kayak(
            start_x=start_x,
            start_y=start_y,
            duration_hours=duration,
            paddling=profile,
        )

        return runner, result

    console.print("[bold]Running simulation...[/bold]")
    runner, result = asyncio.run(_run())

    # Display results
    table = Table(title="Simulation Results")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Duration", f"{result.duration_seconds / 3600:.1f} hours")
    table.add_row("Total distance", f"{result.total_distance_m:.0f} m")
    table.add_row("Net displacement", f"{result.net_displacement_m:.0f} m")
    table.add_row("Mean speed", f"{result.mean_speed_ms:.2f} m/s ({result.mean_speed_ms * 3.6:.1f} km/h)")
    table.add_row("Max speed", f"{result.max_speed_ms:.2f} m/s ({result.max_speed_ms * 3.6:.1f} km/h)")
    table.add_row("Tide phase at start", result.tide_phase_at_start)
    console.print(table)

    # Save output
    if output:
        result.save(output)
        console.print(f"[green]Results saved to {output}[/green]")

    # Show plot
    if plot:
        from waitangi.viz.plots import plot_trajectory
        import matplotlib.pyplot as plt

        fig = plot_trajectory(result, mesh=runner.mesh)
        plt.show()


@app.command()
def particles(
    n_particles: Annotated[int, typer.Option(help="Number of particles")] = 1000,
    duration: Annotated[float, typer.Option(help="Duration in hours")] = 2.0,
    start_chainage: Annotated[float, typer.Option(help="Start distance from mouth (m)")] = 500.0,
    spread: Annotated[float, typer.Option(help="Initial spread radius (m)")] = 20.0,
    plot: Annotated[bool, typer.Option(help="Show evolution plot")] = True,
):
    """Run a particle cloud simulation."""
    from waitangi.simulation.runner import SimulationRunner
    from waitangi.viz.plots import plot_particle_evolution

    async def _run():
        runner = SimulationRunner()
        await runner.initialize(use_synthetic_data=True)

        # Get start position
        start_x, start_y = runner.mesh.chainage_to_point(start_chainage)

        # Run simulation
        history = runner.run_particle_cloud(
            start_x=start_x,
            start_y=start_y,
            n_particles=n_particles,
            spread_m=spread,
            duration_hours=duration,
        )

        return runner, history

    console.print(f"[bold]Running {n_particles} particles...[/bold]")
    runner, history = asyncio.run(_run())

    # Display statistics
    stats = runner.velocity_field.mesh  # Get final statistics
    console.print(f"Simulation complete. {len(history)} snapshots recorded.")

    if plot:
        import matplotlib.pyplot as plt
        fig = plot_particle_evolution(history, mesh=runner.mesh)
        plt.show()


@app.command()
def profile(
    output: Annotated[Optional[Path], typer.Option(help="Output plot file")] = None,
):
    """Show velocity profile along river."""
    from waitangi.simulation.runner import SimulationRunner
    from waitangi.viz.plots import plot_velocity_profile

    async def _run():
        runner = SimulationRunner()
        await runner.initialize(use_synthetic_data=True)
        profile = runner.velocity_field.get_velocity_profile(datetime.now())
        return runner, profile

    runner, profile_data = asyncio.run(_run())

    # Console summary
    console.print(Panel.fit(
        f"[bold]Velocity Profile[/bold]\n"
        f"River length: {float(profile_data['chainage_m'][-1]):.0f} m\n"
        f"Cancellation zone: {profile_data['cancellation_m']:.0f} m from mouth"
    ))

    import matplotlib.pyplot as plt
    fig = plot_velocity_profile(profile_data, save_path=output)
    if output is None:
        plt.show()
    else:
        console.print(f"[green]Profile saved to {output}[/green]")


@app.command()
def dashboard(
    hours: Annotated[int, typer.Option(help="Hours to forecast")] = 12,
    output: Annotated[Optional[Path], typer.Option(help="Output plot file")] = None,
):
    """Show conditions dashboard."""
    from waitangi.simulation.runner import SimulationRunner
    from waitangi.viz.plots import plot_conditions_dashboard

    async def _run():
        runner = SimulationRunner()
        await runner.initialize(use_synthetic_data=True)
        return runner

    runner = asyncio.run(_run())

    import matplotlib.pyplot as plt
    fig = plot_conditions_dashboard(
        runner.tide_model,
        runner.river_model,
        runner.wind_model,
        hours_ahead=hours,
        save_path=output,
    )

    if output is None:
        plt.show()
    else:
        console.print(f"[green]Dashboard saved to {output}[/green]")


@app.command()
def export_geojson(
    output: Annotated[Path, typer.Argument(help="Output GeoJSON file")],
    include_mesh: Annotated[bool, typer.Option(help="Include mesh triangles")] = False,
):
    """Export river geometry as GeoJSON."""
    from waitangi.models.geometry import create_river_mesh
    from waitangi.viz.maps import create_river_map
    import json

    mesh = create_river_mesh()
    geojson = create_river_map(mesh, include_triangles=include_mesh)

    with open(output, "w") as f:
        json.dump(geojson, f, indent=2)

    console.print(f"[green]GeoJSON exported to {output}[/green]")
    console.print(f"River length: {mesh.river_length:.0f} m")
    console.print(f"Mesh: {mesh.n_nodes} nodes, {mesh.n_triangles} triangles")


@app.command()
def animate(
    output: Annotated[Path, typer.Argument(help="Output video file (e.g. kayak.mp4)")] = Path("kayak.mp4"),
    duration: Annotated[float, typer.Option(help="Simulation duration in hours")] = 2.0,
    fps: Annotated[int, typer.Option(help="Video frame rate")] = 30,
    paddle_speed: Annotated[float, typer.Option(help="Paddling speed km/h")] = 0.0,
    direction: Annotated[str, typer.Option(help="Direction: upstream/downstream")] = "upstream",
    start_chainage: Annotated[float, typer.Option(help="Start distance from mouth (m)")] = 500.0,
    width: Annotated[int, typer.Option(help="Video width")] = 1280,
    height: Annotated[int, typer.Option(help="Video height")] = 720,
    play: Annotated[bool, typer.Option(help="Open video after creation")] = True,
):
    """Create animated video of kayak simulation."""
    from waitangi.viz.animate import animate_kayak_simulation

    console.print(f"[bold]Creating kayak animation...[/bold]")
    console.print(f"  Duration: {duration} hours")
    console.print(f"  Paddling: {paddle_speed} km/h {direction if paddle_speed > 0 else '(drift)'}")
    console.print(f"  Output: {output}")

    animate_kayak_simulation(
        output_path=output,
        duration_hours=duration,
        fps=fps,
        width=width,
        height=height,
        paddle_speed_kmh=paddle_speed,
        direction=direction,
        start_chainage=start_chainage,
    )

    console.print(f"[green]Video created: {output}[/green]")

    if play:
        import subprocess
        import sys
        if sys.platform == "darwin":
            subprocess.run(["open", str(output)])
        elif sys.platform == "linux":
            subprocess.run(["xdg-open", str(output)])
        elif sys.platform == "win32":
            subprocess.run(["start", str(output)], shell=True)


@app.command()
def version():
    """Show version information."""
    from waitangi import __version__
    console.print(f"waitangi-water-models v{__version__}")


if __name__ == "__main__":
    app()
