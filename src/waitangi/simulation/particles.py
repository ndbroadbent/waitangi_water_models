"""GPU-accelerated particle system for bulk kayak advection.

Implements Lagrangian particle advection through an Eulerian velocity field
using JAX for GPU acceleration.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial

import jax.numpy as jnp
from jax import Array, jit, lax, vmap

from waitangi.core.config import SimulationSettings, get_settings


@dataclass
class ParticleState:
    """State of all particles at a point in time.

    All arrays have shape (N,) where N is number of particles.
    Uses structure-of-arrays layout for GPU efficiency.
    """

    # Positions (meters, NZTM)
    x: Array
    y: Array

    # Velocities (m/s)
    vx: Array
    vy: Array

    # Paddling velocities (m/s, relative to water)
    paddle_vx: Array
    paddle_vy: Array

    # Active flags (1.0 = active, 0.0 = inactive/boundary)
    active: Array

    # Timestamp
    timestamp: datetime

    @property
    def n_particles(self) -> int:
        """Number of particles."""
        return int(self.x.shape[0])

    @property
    def n_active(self) -> int:
        """Number of active particles."""
        return int(jnp.sum(self.active))


class ParticleSystem:
    """GPU-accelerated particle advection system.

    Supports:
    - Batch advection of thousands of particles
    - Euler and RK2 integration
    - Boundary handling (deactivate particles leaving domain)
    """

    def __init__(
        self,
        n_particles: int,
        settings: SimulationSettings | None = None,
    ):
        """Initialize particle system.

        Args:
            n_particles: Number of particles to simulate.
            settings: Simulation settings.
        """
        self.n_particles = n_particles
        self.settings = settings or get_settings().simulation

        self.state: ParticleState | None = None
        self.history: list[ParticleState] = []

    def initialize_at_point(
        self,
        x: float,
        y: float,
        spread: float = 10.0,
        timestamp: datetime | None = None,
    ) -> ParticleState:
        """Initialize all particles near a single point.

        Args:
            x: Center x coordinate.
            y: Center y coordinate.
            spread: Random spread radius (meters).
            timestamp: Initial timestamp.

        Returns:
            Initial particle state.
        """
        import jax.random as random

        key = random.PRNGKey(42)
        key1, key2 = random.split(key)

        # Random positions in circle around center
        angles = random.uniform(key1, (self.n_particles,)) * 2 * jnp.pi
        radii = random.uniform(key2, (self.n_particles,)) * spread

        x_arr = jnp.full(self.n_particles, x) + radii * jnp.cos(angles)
        y_arr = jnp.full(self.n_particles, y) + radii * jnp.sin(angles)

        self.state = ParticleState(
            x=x_arr,
            y=y_arr,
            vx=jnp.zeros(self.n_particles),
            vy=jnp.zeros(self.n_particles),
            paddle_vx=jnp.zeros(self.n_particles),
            paddle_vy=jnp.zeros(self.n_particles),
            active=jnp.ones(self.n_particles),
            timestamp=timestamp or datetime.now(),
        )

        self.history = [self.state]
        return self.state

    def initialize_along_line(
        self,
        x_start: float,
        y_start: float,
        x_end: float,
        y_end: float,
        timestamp: datetime | None = None,
    ) -> ParticleState:
        """Initialize particles along a line.

        Useful for visualizing flow across a transect.

        Args:
            x_start, y_start: Start of line.
            x_end, y_end: End of line.
            timestamp: Initial timestamp.

        Returns:
            Initial particle state.
        """
        t = jnp.linspace(0, 1, self.n_particles)
        x_arr = x_start + t * (x_end - x_start)
        y_arr = y_start + t * (y_end - y_start)

        self.state = ParticleState(
            x=x_arr,
            y=y_arr,
            vx=jnp.zeros(self.n_particles),
            vy=jnp.zeros(self.n_particles),
            paddle_vx=jnp.zeros(self.n_particles),
            paddle_vy=jnp.zeros(self.n_particles),
            active=jnp.ones(self.n_particles),
            timestamp=timestamp or datetime.now(),
        )

        self.history = [self.state]
        return self.state

    def set_paddling(
        self,
        paddle_speed: float,
        direction_x: Array,
        direction_y: Array,
    ) -> None:
        """Set paddling velocity for all particles.

        Args:
            paddle_speed: Paddling speed (m/s).
            direction_x: Unit vector x-component for each particle.
            direction_y: Unit vector y-component for each particle.
        """
        if self.state is None:
            raise ValueError("Particles not initialized")

        self.state = ParticleState(
            x=self.state.x,
            y=self.state.y,
            vx=self.state.vx,
            vy=self.state.vy,
            paddle_vx=paddle_speed * direction_x,
            paddle_vy=paddle_speed * direction_y,
            active=self.state.active,
            timestamp=self.state.timestamp,
        )

    @partial(jit, static_argnums=(0,))
    def _step_euler(
        self,
        x: Array,
        y: Array,
        water_u: Array,
        water_v: Array,
        wind_u: Array,
        wind_v: Array,
        paddle_u: Array,
        paddle_v: Array,
        active: Array,
        dt: float,
    ) -> tuple[Array, Array, Array, Array]:
        """Single Euler integration step (JIT-compiled).

        Returns:
            (new_x, new_y, new_vx, new_vy)
        """
        # Total velocity
        vx = water_u + wind_u + paddle_u
        vy = water_v + wind_v + paddle_v

        # Update positions (only for active particles)
        new_x = x + vx * dt * active
        new_y = y + vy * dt * active

        return new_x, new_y, vx, vy

    def step(
        self,
        water_velocity_fn: callable,
        wind_drift_fn: callable,
        dt: float | None = None,
    ) -> ParticleState:
        """Advance particles by one time step.

        Args:
            water_velocity_fn: Function (x, y) -> (u, v) for water velocity.
            wind_drift_fn: Function (x, y) -> (u, v) for wind drift.
            dt: Time step (seconds). Uses settings default if None.

        Returns:
            New particle state.
        """
        if self.state is None:
            raise ValueError("Particles not initialized")

        if dt is None:
            dt = self.settings.dt

        # Get velocities at current positions
        water_u, water_v = water_velocity_fn(self.state.x, self.state.y)
        wind_u, wind_v = wind_drift_fn(self.state.x, self.state.y)

        # Euler step
        new_x, new_y, new_vx, new_vy = self._step_euler(
            self.state.x,
            self.state.y,
            water_u,
            water_v,
            wind_u,
            wind_v,
            self.state.paddle_vx,
            self.state.paddle_vy,
            self.state.active,
            dt,
        )

        # Update timestamp
        new_timestamp = self.state.timestamp + timedelta(seconds=dt)

        self.state = ParticleState(
            x=new_x,
            y=new_y,
            vx=new_vx,
            vy=new_vy,
            paddle_vx=self.state.paddle_vx,
            paddle_vy=self.state.paddle_vy,
            active=self.state.active,
            timestamp=new_timestamp,
        )

        return self.state

    def step_rk2(
        self,
        water_velocity_fn: callable,
        wind_drift_fn: callable,
        dt: float | None = None,
    ) -> ParticleState:
        """Advance particles using 2nd-order Runge-Kutta.

        Args:
            water_velocity_fn: Function (x, y) -> (u, v) for water velocity.
            wind_drift_fn: Function (x, y) -> (u, v) for wind drift.
            dt: Time step (seconds).

        Returns:
            New particle state.
        """
        if self.state is None:
            raise ValueError("Particles not initialized")

        if dt is None:
            dt = self.settings.dt

        x0, y0 = self.state.x, self.state.y
        paddle_u, paddle_v = self.state.paddle_vx, self.state.paddle_vy
        active = self.state.active

        # K1: Velocity at current position
        water_u1, water_v1 = water_velocity_fn(x0, y0)
        wind_u1, wind_v1 = wind_drift_fn(x0, y0)
        k1_x = water_u1 + wind_u1 + paddle_u
        k1_y = water_v1 + wind_v1 + paddle_v

        # Midpoint position
        x_mid = x0 + k1_x * dt / 2 * active
        y_mid = y0 + k1_y * dt / 2 * active

        # K2: Velocity at midpoint
        water_u2, water_v2 = water_velocity_fn(x_mid, y_mid)
        wind_u2, wind_v2 = wind_drift_fn(x_mid, y_mid)
        k2_x = water_u2 + wind_u2 + paddle_u
        k2_y = water_v2 + wind_v2 + paddle_v

        # Update using K2
        new_x = x0 + k2_x * dt * active
        new_y = y0 + k2_y * dt * active

        # Update timestamp
        new_timestamp = self.state.timestamp + timedelta(seconds=dt)

        self.state = ParticleState(
            x=new_x,
            y=new_y,
            vx=k2_x,
            vy=k2_y,
            paddle_vx=paddle_u,
            paddle_vy=paddle_v,
            active=active,
            timestamp=new_timestamp,
        )

        return self.state

    def run(
        self,
        water_velocity_fn: callable,
        wind_drift_fn: callable,
        duration_seconds: float,
        dt: float | None = None,
        record_interval: float | None = None,
    ) -> list[ParticleState]:
        """Run simulation for specified duration.

        Args:
            water_velocity_fn: Function (x, y) -> (u, v).
            wind_drift_fn: Function (x, y) -> (u, v).
            duration_seconds: Total simulation time.
            dt: Time step (seconds).
            record_interval: Interval for recording history (seconds).

        Returns:
            List of recorded states.
        """
        if dt is None:
            dt = self.settings.dt
        if record_interval is None:
            record_interval = self.settings.output_interval

        n_steps = int(duration_seconds / dt)
        record_every = max(1, int(record_interval / dt))

        use_rk2 = self.settings.use_rk2

        for i in range(n_steps):
            if use_rk2:
                self.step_rk2(water_velocity_fn, wind_drift_fn, dt)
            else:
                self.step(water_velocity_fn, wind_drift_fn, dt)

            if i % record_every == 0:
                self.history.append(self.state)

        return self.history

    def apply_boundary(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> int:
        """Deactivate particles outside boundary.

        Args:
            x_min, x_max: X bounds.
            y_min, y_max: Y bounds.

        Returns:
            Number of particles deactivated.
        """
        if self.state is None:
            return 0

        in_bounds = (
            (self.state.x >= x_min)
            & (self.state.x <= x_max)
            & (self.state.y >= y_min)
            & (self.state.y <= y_max)
        )

        old_active = self.state.n_active
        new_active = self.state.active * in_bounds.astype(jnp.float32)

        self.state = ParticleState(
            x=self.state.x,
            y=self.state.y,
            vx=self.state.vx,
            vy=self.state.vy,
            paddle_vx=self.state.paddle_vx,
            paddle_vy=self.state.paddle_vy,
            active=new_active,
            timestamp=self.state.timestamp,
        )

        return old_active - self.state.n_active

    def get_statistics(self) -> dict:
        """Get statistics about current particle distribution.

        Returns:
            Dictionary with mean position, spread, etc.
        """
        if self.state is None:
            return {}

        active = self.state.active
        n_active = jnp.sum(active)

        if n_active < 1:
            return {"n_active": 0}

        # Weighted statistics (only active particles)
        mean_x = jnp.sum(self.state.x * active) / n_active
        mean_y = jnp.sum(self.state.y * active) / n_active
        mean_speed = jnp.sum(
            jnp.sqrt(self.state.vx**2 + self.state.vy**2) * active
        ) / n_active

        # Spread (standard deviation)
        var_x = jnp.sum((self.state.x - mean_x)**2 * active) / n_active
        var_y = jnp.sum((self.state.y - mean_y)**2 * active) / n_active

        return {
            "n_active": int(n_active),
            "mean_x": float(mean_x),
            "mean_y": float(mean_y),
            "std_x": float(jnp.sqrt(var_x)),
            "std_y": float(jnp.sqrt(var_y)),
            "mean_speed_ms": float(mean_speed),
            "timestamp": self.state.timestamp,
        }
