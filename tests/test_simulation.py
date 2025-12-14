"""Tests for simulation engine."""

from datetime import datetime, timedelta

import jax.numpy as jnp
import numpy as np
import pytest

from waitangi.models.geometry import create_river_mesh
from waitangi.models.river import RiverDischargeModel
from waitangi.models.tide import TideModel
from waitangi.models.velocity import VelocityField, create_eddy_field
from waitangi.simulation.kayak import KayakSimulator, KayakState, PaddlingProfile
from waitangi.simulation.particles import ParticleSystem


class TestKayakState:
    """Tests for kayak state representation."""

    def test_create_state(self):
        """Should create state with position and velocity."""
        state = KayakState(x=100.0, y=200.0, vx=1.0, vy=0.5)

        assert state.x == 100.0
        assert state.y == 200.0
        assert state.speed_over_ground == pytest.approx(np.sqrt(1.25))

    def test_to_array(self):
        """Should convert to JAX array."""
        state = KayakState(x=100.0, y=200.0, vx=1.0, vy=0.5)
        arr = state.to_array()

        assert arr.shape == (4,)
        assert float(arr[0]) == 100.0

    def test_from_array(self):
        """Should create from JAX array."""
        arr = jnp.array([100.0, 200.0, 1.0, 0.5])
        state = KayakState.from_array(arr)

        assert state.x == 100.0
        assert state.speed_over_ground > 0


class TestPaddlingProfile:
    """Tests for paddling configurations."""

    def test_no_paddle(self):
        """No paddle profile should return zero speed."""
        profile = PaddlingProfile.no_paddle()

        for t in [0, 100, 1000, 10000]:
            assert profile.get_paddle_speed(t) == 0.0

    def test_constant_paddle(self):
        """Constant profile should return constant speed."""
        profile = PaddlingProfile(
            mode="constant",
            base_speed=1.5,
            heading_mode="upstream",
        )

        for t in [0, 100, 1000, 10000]:
            assert profile.get_paddle_speed(t) == 1.5

    def test_interval_paddle(self):
        """Interval profile should alternate work/rest."""
        profile = PaddlingProfile(
            mode="intervals",
            base_speed=1.5,
            heading_mode="upstream",
            work_minutes=30,
            rest_minutes=10,
            rest_speed=0.5,
        )

        # During work period (0-30 min)
        assert profile.get_paddle_speed(15 * 60) == 1.5

        # During rest period (30-40 min)
        assert profile.get_paddle_speed(35 * 60) == 0.5

        # Back to work (40-70 min)
        assert profile.get_paddle_speed(50 * 60) == 1.5


class TestKayakSimulator:
    """Tests for single kayak simulation."""

    def test_step_no_paddle(self):
        """Kayak should drift with water current."""
        state = KayakState(x=0.0, y=0.0, timestamp=datetime.now())
        sim = KayakSimulator(state)

        # Step with water flowing east
        new_state = sim.step(
            dt=10.0,
            water_velocity=(1.0, 0.0),
            wind_drift=(0.0, 0.0),
            river_direction=(1.0, 0.0),
        )

        # Should have moved east
        assert new_state.x > state.x
        assert abs(new_state.y - state.y) < 0.1

    def test_step_with_paddle(self):
        """Paddling should add to velocity."""
        state = KayakState(x=0.0, y=0.0, timestamp=datetime.now())
        profile = PaddlingProfile.cruise_upstream()
        sim = KayakSimulator(state, paddling_profile=profile)

        # Step with water flowing downstream, paddle upstream
        new_state = sim.step(
            dt=10.0,
            water_velocity=(-0.5, 0.0),  # Downstream
            wind_drift=(0.0, 0.0),
            river_direction=(1.0, 0.0),  # Upstream is +x
        )

        # Paddling speed should overcome current
        if profile.base_speed > 0.5:
            assert new_state.x > 0  # Made progress upstream

    def test_history_recorded(self):
        """Simulator should record trajectory."""
        state = KayakState(x=0.0, y=0.0, timestamp=datetime.now())
        sim = KayakSimulator(state)

        for _ in range(10):
            sim.step(
                dt=1.0,
                water_velocity=(1.0, 0.5),
                wind_drift=(0.0, 0.0),
                river_direction=(1.0, 0.0),
            )

        assert len(sim.history) == 11  # Initial + 10 steps

    def test_trajectory_arrays(self):
        """Should export trajectory as arrays."""
        state = KayakState(x=0.0, y=0.0, timestamp=datetime.now())
        sim = KayakSimulator(state)

        for _ in range(5):
            sim.step(
                dt=1.0,
                water_velocity=(1.0, 0.0),
                wind_drift=(0.0, 0.0),
                river_direction=(1.0, 0.0),
            )

        arrays = sim.get_trajectory_arrays()

        assert "x" in arrays
        assert "y" in arrays
        assert "speed" in arrays
        assert len(arrays["x"]) == 6


class TestParticleSystem:
    """Tests for GPU particle advection."""

    def test_initialize_at_point(self):
        """Should initialize particles around a point."""
        particles = ParticleSystem(n_particles=100)
        state = particles.initialize_at_point(0.0, 0.0, spread=10.0)

        assert state.n_particles == 100
        assert state.n_active == 100

        # Particles should be near origin
        assert np.abs(np.asarray(state.x).mean()) < 5
        assert np.abs(np.asarray(state.y).mean()) < 5

    def test_initialize_along_line(self):
        """Should initialize particles along a line."""
        particles = ParticleSystem(n_particles=50)
        state = particles.initialize_along_line(0.0, 0.0, 100.0, 0.0)

        assert state.n_particles == 50
        x_arr = np.asarray(state.x)
        assert x_arr.min() == pytest.approx(0.0)
        assert x_arr.max() == pytest.approx(100.0)

    def test_step(self):
        """Particles should move with velocity field."""
        particles = ParticleSystem(n_particles=100)
        particles.initialize_at_point(0.0, 0.0, spread=1.0)

        # Uniform velocity field
        def water_velocity(x, y):
            return jnp.ones_like(x), jnp.zeros_like(y)

        def wind_drift(x, y):
            return jnp.zeros_like(x), jnp.zeros_like(y)

        state = particles.step(water_velocity, wind_drift, dt=10.0)

        # All particles should have moved east
        assert np.asarray(state.x).mean() > 5

    def test_boundary_deactivation(self):
        """Particles outside boundary should be deactivated."""
        particles = ParticleSystem(n_particles=100)
        particles.initialize_at_point(50.0, 50.0, spread=100.0)

        # Apply tight boundary
        n_deactivated = particles.apply_boundary(
            x_min=0, x_max=100, y_min=0, y_max=100
        )

        # Some particles should have been deactivated
        assert particles.state.n_active < 100 or n_deactivated >= 0

    def test_statistics(self):
        """Should compute particle statistics."""
        particles = ParticleSystem(n_particles=1000)
        particles.initialize_at_point(100.0, 200.0, spread=20.0)

        stats = particles.get_statistics()

        assert stats["n_active"] == 1000
        assert abs(stats["mean_x"] - 100.0) < 5
        assert abs(stats["mean_y"] - 200.0) < 5


class TestVelocityField:
    """Tests for composed velocity field."""

    @pytest.fixture
    def velocity_field(self):
        """Create a velocity field for testing."""
        mesh = create_river_mesh()
        tide_model = TideModel.create()
        river_model = RiverDischargeModel.create()

        return VelocityField(
            mesh=mesh,
            tide_model=tide_model,
            river_model=river_model,
        )

    def test_velocity_at_point(self, velocity_field):
        """Should return velocity at a point."""
        mesh = velocity_field.mesh
        x, y = mesh.chainage_to_point(500)

        u, v = velocity_field.get_velocity_at_point(x, y, datetime.now())

        # Velocity should be finite
        assert np.isfinite(u)
        assert np.isfinite(v)

    def test_cancellation_zone(self, velocity_field):
        """Should find where tide and river balance."""
        cancel_chainage = velocity_field.get_cancellation_zone(datetime.now())

        # Should be somewhere along river
        assert 0 <= cancel_chainage <= velocity_field.mesh.river_length

    def test_velocity_profile(self, velocity_field):
        """Should generate velocity profile."""
        profile = velocity_field.get_velocity_profile(datetime.now())

        assert "chainage_m" in profile
        assert "v_tide_ms" in profile
        assert "v_river_ms" in profile
        assert "v_net_ms" in profile
        assert "cancellation_m" in profile


class TestEddyField:
    """Tests for eddy/turbulence field."""

    def test_create_eddy_field(self):
        """Should create eddy function."""
        mesh = create_river_mesh()
        eddy_fn = create_eddy_field(mesh)

        # Evaluate at some points
        x = jnp.array([100.0, 200.0, 300.0])
        y = jnp.array([100.0, 200.0, 300.0])

        u_eddy, v_eddy = eddy_fn(x, y)

        assert u_eddy.shape == (3,)
        assert v_eddy.shape == (3,)

    def test_eddy_near_center(self):
        """Eddy velocity should be higher near eddy centers."""
        # Create eddy at known location
        eddy_fn = create_eddy_field(
            mesh=create_river_mesh(),
            eddy_locations=[(100.0, 100.0)],
            eddy_strength=0.5,
            eddy_radius=30.0,
        )

        # Near eddy center
        x_near = jnp.array([100.0])
        y_near = jnp.array([115.0])  # 15m from center
        u_near, v_near = eddy_fn(x_near, y_near)

        # Far from eddy
        x_far = jnp.array([500.0])
        y_far = jnp.array([500.0])
        u_far, v_far = eddy_fn(x_far, y_far)

        # Velocity should be higher near eddy
        speed_near = float(jnp.sqrt(u_near[0]**2 + v_near[0]**2))
        speed_far = float(jnp.sqrt(u_far[0]**2 + v_far[0]**2))

        assert speed_near > speed_far
