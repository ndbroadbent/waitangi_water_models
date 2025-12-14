"""Kayak state and dynamics model.

A kayak is a Lagrangian particle advected through the Eulerian velocity field,
with additional paddling and wind forcing.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import jax.numpy as jnp
import numpy as np
from jax import Array

from waitangi.core.config import KayakSettings, get_settings


@dataclass
class KayakState:
    """State of a kayak at a point in time.

    Position in NZTM2000 coordinates (meters).
    Velocity in m/s.
    """

    # Position
    x: float
    y: float

    # Velocity (ground-relative)
    vx: float = 0.0
    vy: float = 0.0

    # Timestamp
    timestamp: datetime | None = None

    # Paddling state
    paddle_speed: float = 0.0  # m/s relative to water
    paddle_heading: float = 0.0  # degrees from north

    # Metadata
    chainage: float = 0.0  # distance from mouth (m)

    @property
    def speed_over_ground(self) -> float:
        """Speed over ground in m/s."""
        return float(np.sqrt(self.vx**2 + self.vy**2))

    def to_array(self) -> Array:
        """Convert to JAX array [x, y, vx, vy]."""
        return jnp.array([self.x, self.y, self.vx, self.vy])

    @classmethod
    def from_array(cls, arr: Array, timestamp: datetime | None = None) -> "KayakState":
        """Create from JAX array."""
        return cls(
            x=float(arr[0]),
            y=float(arr[1]),
            vx=float(arr[2]),
            vy=float(arr[3]),
            timestamp=timestamp,
        )


@dataclass
class PaddlingProfile:
    """Paddling effort profile over time.

    Can represent constant effort, scheduled breaks, or variable intensity.
    """

    mode: Literal["constant", "intervals", "none"]
    base_speed: float  # m/s
    heading_mode: Literal["upstream", "downstream", "fixed"]
    fixed_heading: float = 0.0  # degrees, if heading_mode == "fixed"

    # For interval mode
    work_minutes: float = 30.0
    rest_minutes: float = 5.0
    rest_speed: float = 0.0

    def get_paddle_speed(self, elapsed_seconds: float) -> float:
        """Get paddling speed at given time since start."""
        if self.mode == "none":
            return 0.0
        if self.mode == "constant":
            return self.base_speed

        # Interval mode
        cycle_seconds = (self.work_minutes + self.rest_minutes) * 60
        position_in_cycle = elapsed_seconds % cycle_seconds

        if position_in_cycle < self.work_minutes * 60:
            return self.base_speed
        return self.rest_speed

    @classmethod
    def no_paddle(cls) -> "PaddlingProfile":
        """Create a drifting (no paddle) profile."""
        return cls(mode="none", base_speed=0.0, heading_mode="downstream")

    @classmethod
    def cruise_upstream(cls, settings: KayakSettings | None = None) -> "PaddlingProfile":
        """Create a steady upstream paddling profile."""
        if settings is None:
            settings = get_settings().kayak
        return cls(
            mode="constant",
            base_speed=settings.cruise_paddle_speed,
            heading_mode="upstream",
        )

    @classmethod
    def cruise_downstream(cls, settings: KayakSettings | None = None) -> "PaddlingProfile":
        """Create a steady downstream paddling profile."""
        if settings is None:
            settings = get_settings().kayak
        return cls(
            mode="constant",
            base_speed=settings.cruise_paddle_speed,
            heading_mode="downstream",
        )


class KayakSimulator:
    """Single-kayak simulator with detailed state tracking."""

    def __init__(
        self,
        initial_state: KayakState,
        paddling_profile: PaddlingProfile | None = None,
        settings: KayakSettings | None = None,
    ):
        """Initialize kayak simulator.

        Args:
            initial_state: Starting position and state.
            paddling_profile: Paddling effort profile.
            settings: Kayak configuration.
        """
        self.state = initial_state
        self.profile = paddling_profile or PaddlingProfile.no_paddle()
        self.settings = settings or get_settings().kayak

        self.start_time = initial_state.timestamp or datetime.now()
        self.history: list[KayakState] = [initial_state]

    def step(
        self,
        dt: float,
        water_velocity: tuple[float, float],
        wind_drift: tuple[float, float],
        river_direction: tuple[float, float],
    ) -> KayakState:
        """Advance kayak by one time step.

        Args:
            dt: Time step in seconds.
            water_velocity: (u, v) water velocity in m/s.
            wind_drift: (u, v) wind-induced drift in m/s.
            river_direction: Unit vector (dx, dy) pointing upstream.

        Returns:
            New kayak state.
        """
        elapsed = (
            (self.state.timestamp - self.start_time).total_seconds()
            if self.state.timestamp
            else 0.0
        )

        # Get paddling velocity
        paddle_speed = self.profile.get_paddle_speed(elapsed)
        paddle_u, paddle_v = self._get_paddle_velocity(paddle_speed, river_direction)

        # Total velocity
        vx = water_velocity[0] + wind_drift[0] + paddle_u
        vy = water_velocity[1] + wind_drift[1] + paddle_v

        # Update position (simple Euler)
        new_x = self.state.x + vx * dt
        new_y = self.state.y + vy * dt

        # Update timestamp
        new_timestamp = None
        if self.state.timestamp:
            new_timestamp = self.state.timestamp + timedelta(seconds=dt)

        new_state = KayakState(
            x=new_x,
            y=new_y,
            vx=vx,
            vy=vy,
            timestamp=new_timestamp,
            paddle_speed=paddle_speed,
            paddle_heading=self.state.paddle_heading,
        )

        self.state = new_state
        self.history.append(new_state)

        return new_state

    def step_rk2(
        self,
        dt: float,
        get_velocities: callable,
    ) -> KayakState:
        """Advance kayak using 2nd-order Runge-Kutta.

        Args:
            dt: Time step in seconds.
            get_velocities: Function (x, y, t) -> (water_uv, wind_uv, river_dir).

        Returns:
            New kayak state.
        """
        x0, y0 = self.state.x, self.state.y
        t0 = self.state.timestamp or self.start_time

        # K1: Velocity at current position
        water1, wind1, dir1 = get_velocities(x0, y0, t0)
        paddle1 = self._get_paddle_velocity(
            self.profile.get_paddle_speed(
                (t0 - self.start_time).total_seconds()
            ),
            dir1,
        )
        k1_x = water1[0] + wind1[0] + paddle1[0]
        k1_y = water1[1] + wind1[1] + paddle1[1]

        # K2: Velocity at midpoint
        t_mid = t0 + timedelta(seconds=dt / 2)
        x_mid = x0 + k1_x * dt / 2
        y_mid = y0 + k1_y * dt / 2

        water2, wind2, dir2 = get_velocities(x_mid, y_mid, t_mid)
        paddle2 = self._get_paddle_velocity(
            self.profile.get_paddle_speed(
                (t_mid - self.start_time).total_seconds()
            ),
            dir2,
        )
        k2_x = water2[0] + wind2[0] + paddle2[0]
        k2_y = water2[1] + wind2[1] + paddle2[1]

        # Update position using K2
        new_x = x0 + k2_x * dt
        new_y = y0 + k2_y * dt
        new_timestamp = t0 + timedelta(seconds=dt)

        new_state = KayakState(
            x=new_x,
            y=new_y,
            vx=k2_x,
            vy=k2_y,
            timestamp=new_timestamp,
            paddle_speed=self.profile.get_paddle_speed(
                (new_timestamp - self.start_time).total_seconds()
            ),
        )

        self.state = new_state
        self.history.append(new_state)

        return new_state

    def _get_paddle_velocity(
        self,
        speed: float,
        river_direction: tuple[float, float],
    ) -> tuple[float, float]:
        """Convert paddle speed to velocity components.

        Args:
            speed: Paddling speed in m/s.
            river_direction: Unit vector pointing upstream.

        Returns:
            (u, v) paddling velocity in m/s.
        """
        if speed < 0.01:
            return 0.0, 0.0

        if self.profile.heading_mode == "upstream":
            # Paddle in upstream direction
            return speed * river_direction[0], speed * river_direction[1]
        elif self.profile.heading_mode == "downstream":
            # Paddle in downstream direction (opposite to river_direction)
            return -speed * river_direction[0], -speed * river_direction[1]
        else:
            # Fixed heading
            heading_rad = np.radians(90 - self.profile.fixed_heading)
            return speed * np.cos(heading_rad), speed * np.sin(heading_rad)

    def get_trajectory_arrays(self) -> dict[str, np.ndarray]:
        """Export trajectory as numpy arrays for analysis.

        Returns:
            Dictionary with x, y, t, speed arrays.
        """
        n = len(self.history)
        return {
            "x": np.array([s.x for s in self.history]),
            "y": np.array([s.y for s in self.history]),
            "vx": np.array([s.vx for s in self.history]),
            "vy": np.array([s.vy for s in self.history]),
            "speed": np.array([s.speed_over_ground for s in self.history]),
            "paddle_speed": np.array([s.paddle_speed for s in self.history]),
            "timestamp": np.array([
                s.timestamp.timestamp() if s.timestamp else 0.0
                for s in self.history
            ]),
        }
