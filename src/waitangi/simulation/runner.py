"""Simulation orchestration and result handling."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel

from waitangi.core.config import Settings, get_settings
from waitangi.data.gauge import GaugeData, fetch_nrc_gauge_data
from waitangi.data.rainfall import RainfallData, fetch_rainfall_data
from waitangi.data.tide import fetch_tide_predictions
from waitangi.data.weather import WeatherData, fetch_weather_forecast
from waitangi.models.geometry import Mesh, create_river_mesh
from waitangi.models.river import RiverDischargeModel
from waitangi.models.tide import TideModel
from waitangi.models.velocity import VelocityField, create_eddy_field
from waitangi.models.wind import WindModel
from waitangi.simulation.kayak import KayakSimulator, KayakState, PaddlingProfile
from waitangi.simulation.particles import ParticleState, ParticleSystem


class SimulationResult(BaseModel):
    """Results from a simulation run."""

    class Config:
        arbitrary_types_allowed = True

    # Metadata
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    n_steps: int

    # Trajectory data (serializable)
    trajectory_x: list[float]
    trajectory_y: list[float]
    trajectory_t: list[float]  # seconds since start
    trajectory_speed: list[float]

    # Summary statistics
    total_distance_m: float
    net_displacement_m: float
    mean_speed_ms: float
    max_speed_ms: float

    # Conditions summary
    tide_phase_at_start: str
    mean_wind_speed_ms: float
    mean_river_flow_m3s: float

    def to_dict(self) -> dict:
        """Export to dictionary for JSON serialization."""
        return self.model_dump()

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        import json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


@dataclass
class SimulationRunner:
    """Main simulation orchestrator.

    Coordinates data ingestion, model setup, and simulation execution.
    """

    settings: Settings = field(default_factory=get_settings)

    # Models (initialized lazily)
    mesh: Mesh | None = None
    tide_model: TideModel | None = None
    river_model: RiverDischargeModel | None = None
    wind_model: WindModel | None = None
    velocity_field: VelocityField | None = None

    # Data (cached)
    gauge_data: GaugeData | None = None
    rainfall_data: RainfallData | None = None
    weather_data: WeatherData | None = None

    async def initialize(self, use_synthetic_data: bool = False) -> None:
        """Initialize all models and fetch data.

        Args:
            use_synthetic_data: If True, skip API calls and use synthetic data.
        """
        # Create mesh
        self.mesh = create_river_mesh()

        # Initialize models
        self.tide_model = TideModel.create(self.settings.tide)
        self.river_model = RiverDischargeModel.create(self.settings.river)
        self.wind_model = WindModel.create(self.settings.wind)

        # Fetch external data
        if not use_synthetic_data:
            await self._fetch_data()

        # Set up velocity field
        eddy_fn = create_eddy_field(self.mesh)
        self.velocity_field = VelocityField(
            mesh=self.mesh,
            tide_model=self.tide_model,
            river_model=self.river_model,
            eddy_fn=eddy_fn,
        )

    async def _fetch_data(self) -> None:
        """Fetch all external data sources."""
        import asyncio

        # Fetch in parallel
        gauge_task = fetch_nrc_gauge_data()
        rainfall_task = fetch_rainfall_data()
        weather_task = fetch_weather_forecast()
        tide_task = fetch_tide_predictions()

        results = await asyncio.gather(
            gauge_task,
            rainfall_task,
            weather_task,
            tide_task,
            return_exceptions=True,
        )

        # Process results
        if not isinstance(results[0], Exception):
            self.gauge_data = results[0]
            self.river_model.ingest_gauge_data(self.gauge_data)

        if not isinstance(results[1], Exception):
            self.rainfall_data = results[1]
            self.river_model.ingest_rainfall_data(self.rainfall_data)

        if not isinstance(results[2], Exception):
            self.weather_data = results[2]
            self.wind_model.ingest_weather_data(self.weather_data)

        if not isinstance(results[3], Exception):
            self.tide_model.ingest_predictions(results[3])

    def run_single_kayak(
        self,
        start_x: float,
        start_y: float,
        start_time: datetime | None = None,
        duration_hours: float = 4.0,
        paddling: PaddlingProfile | None = None,
    ) -> SimulationResult:
        """Run simulation for a single kayak.

        Args:
            start_x: Starting x position (NZTM).
            start_y: Starting y position (NZTM).
            start_time: Start time. Defaults to now.
            duration_hours: Simulation duration.
            paddling: Paddling profile. Defaults to no paddling.

        Returns:
            Simulation results.
        """
        if self.velocity_field is None:
            raise ValueError("Runner not initialized. Call initialize() first.")

        start_time = start_time or datetime.now()
        duration_seconds = duration_hours * 3600
        dt = self.settings.simulation.dt

        # Initialize kayak
        initial_state = KayakState(x=start_x, y=start_y, timestamp=start_time)
        simulator = KayakSimulator(
            initial_state=initial_state,
            paddling_profile=paddling or PaddlingProfile.no_paddle(),
            settings=self.settings.kayak,
        )

        # Define velocity callback
        def get_velocities(x, y, t):
            water_uv = self.velocity_field.get_velocity_at_point(x, y, t)
            wind_uv = self.wind_model.get_kayak_drift(t) if self.wind_model else (0, 0)
            chainage = self.mesh.point_to_chainage(x, y)
            river_dir = self.velocity_field._get_river_direction(chainage)
            return water_uv, wind_uv, river_dir

        # Run simulation
        n_steps = int(duration_seconds / dt)
        use_rk2 = self.settings.simulation.use_rk2

        for _ in range(n_steps):
            if use_rk2:
                simulator.step_rk2(dt, get_velocities)
            else:
                water_uv, wind_uv, river_dir = get_velocities(
                    simulator.state.x,
                    simulator.state.y,
                    simulator.state.timestamp,
                )
                simulator.step(dt, water_uv, wind_uv, river_dir)

        # Build results
        traj = simulator.get_trajectory_arrays()

        # Calculate statistics
        dx = np.diff(traj["x"])
        dy = np.diff(traj["y"])
        segment_lengths = np.sqrt(dx**2 + dy**2)
        total_distance = np.sum(segment_lengths)

        net_dx = traj["x"][-1] - traj["x"][0]
        net_dy = traj["y"][-1] - traj["y"][0]
        net_displacement = np.sqrt(net_dx**2 + net_dy**2)

        return SimulationResult(
            start_time=start_time,
            end_time=simulator.state.timestamp,
            duration_seconds=duration_seconds,
            n_steps=n_steps,
            trajectory_x=traj["x"].tolist(),
            trajectory_y=traj["y"].tolist(),
            trajectory_t=(traj["timestamp"] - traj["timestamp"][0]).tolist(),
            trajectory_speed=traj["speed"].tolist(),
            total_distance_m=float(total_distance),
            net_displacement_m=float(net_displacement),
            mean_speed_ms=float(np.mean(traj["speed"])),
            max_speed_ms=float(np.max(traj["speed"])),
            tide_phase_at_start=self.tide_model.get_phase(start_time),
            mean_wind_speed_ms=float(
                self.weather_data.wind_speeds.mean()
                if self.weather_data
                else 0.0
            ),
            mean_river_flow_m3s=float(
                self.gauge_data.flows[~np.isnan(self.gauge_data.flows)].mean()
                if self.gauge_data
                else self.settings.river.q_base
            ),
        )

    def run_particle_cloud(
        self,
        start_x: float,
        start_y: float,
        n_particles: int = 1000,
        spread_m: float = 20.0,
        start_time: datetime | None = None,
        duration_hours: float = 4.0,
        paddle_speed: float = 0.0,
        paddle_direction: Literal["upstream", "downstream"] = "upstream",
    ) -> list[ParticleState]:
        """Run simulation for a cloud of particles.

        Args:
            start_x: Center x position (NZTM).
            start_y: Center y position (NZTM).
            n_particles: Number of particles.
            spread_m: Initial spread radius.
            start_time: Start time.
            duration_hours: Simulation duration.
            paddle_speed: Paddling speed (m/s).
            paddle_direction: Paddling direction.

        Returns:
            List of particle states over time.
        """
        if self.velocity_field is None:
            raise ValueError("Runner not initialized. Call initialize() first.")

        start_time = start_time or datetime.now()
        duration_seconds = duration_hours * 3600

        # Initialize particles
        particles = ParticleSystem(n_particles, self.settings.simulation)
        particles.initialize_at_point(start_x, start_y, spread_m, start_time)

        # Set paddling if specified
        if paddle_speed > 0:
            chainage = self.mesh.point_to_chainage(start_x, start_y)
            dir_x, dir_y = self.velocity_field._get_river_direction(chainage)
            if paddle_direction == "downstream":
                dir_x, dir_y = -dir_x, -dir_y
            particles.set_paddling(
                paddle_speed,
                jnp.full(n_particles, dir_x),
                jnp.full(n_particles, dir_y),
            )

        # Create velocity functions that capture current time
        current_time = [start_time]  # Mutable container

        def water_velocity_fn(x, y):
            return self.velocity_field.get_velocity_at_particles(
                x, y, current_time[0]
            )

        def wind_drift_fn(x, y):
            if self.wind_model:
                return self.wind_model.get_kayak_drift_field(
                    current_time[0], x, y
                )
            return jnp.zeros_like(x), jnp.zeros_like(y)

        # Run simulation
        dt = self.settings.simulation.dt
        record_interval = self.settings.simulation.output_interval
        n_steps = int(duration_seconds / dt)
        record_every = max(1, int(record_interval / dt))

        history = [particles.state]

        for i in range(n_steps):
            # Update current time for velocity functions
            current_time[0] = start_time + timedelta(seconds=i * dt)

            if self.settings.simulation.use_rk2:
                particles.step_rk2(water_velocity_fn, wind_drift_fn, dt)
            else:
                particles.step(water_velocity_fn, wind_drift_fn, dt)

            if i % record_every == 0:
                history.append(particles.state)

        return history

    def get_current_conditions(self) -> dict:
        """Get current environmental conditions summary.

        Returns:
            Dictionary with tide, wind, flow information.
        """
        now = datetime.now()

        conditions = {
            "timestamp": now,
        }

        if self.tide_model:
            conditions["tide"] = {
                "height_m": self.tide_model.get_height(now),
                "velocity_ms": self.tide_model.get_velocity(now),
                "phase": self.tide_model.get_phase(now),
            }

        if self.river_model:
            conditions["river"] = {
                "discharge_m3s": self.river_model.get_discharge(now),
                "velocity_at_mouth_ms": self.river_model.get_velocity_at_mouth(now),
            }

        if self.wind_model:
            speed, direction = self.wind_model.get_wind(now)
            conditions["wind"] = {
                "speed_ms": speed,
                "direction_deg": direction,
                "description": self.wind_model.get_wind_description(now),
            }

        if self.velocity_field:
            conditions["cancellation_zone_m"] = self.velocity_field.get_cancellation_zone(now)

        return conditions
