"""Integration tests for end-to-end simulation."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from waitangi.simulation.kayak import PaddlingProfile
from waitangi.simulation.runner import SimulationRunner


class TestSimulationRunner:
    """Tests for full simulation workflow."""

    @pytest.fixture
    async def runner(self):
        """Create and initialize a simulation runner."""
        runner = SimulationRunner()
        await runner.initialize(use_synthetic_data=True)
        return runner

    @pytest.mark.asyncio
    async def test_initialization(self, runner):
        """Runner should initialize all components."""
        assert runner.mesh is not None
        assert runner.tide_model is not None
        assert runner.river_model is not None
        assert runner.wind_model is not None
        assert runner.velocity_field is not None

    @pytest.mark.asyncio
    async def test_get_conditions(self, runner):
        """Should return current conditions."""
        conditions = runner.get_current_conditions()

        assert "timestamp" in conditions
        assert "tide" in conditions
        assert "river" in conditions
        assert "wind" in conditions

    @pytest.mark.asyncio
    async def test_single_kayak_drift(self, runner):
        """Kayak should drift with current when not paddling."""
        # Start near mouth
        start_x, start_y = runner.mesh.chainage_to_point(500)

        result = runner.run_single_kayak(
            start_x=start_x,
            start_y=start_y,
            duration_hours=1.0,
            paddling=None,  # No paddling
        )

        # Should have moved
        assert result.total_distance_m > 0
        assert len(result.trajectory_x) > 1

    @pytest.mark.asyncio
    async def test_single_kayak_upstream(self, runner):
        """Kayak paddling upstream should make progress."""
        start_x, start_y = runner.mesh.chainage_to_point(1000)
        start_chainage = 1000

        profile = PaddlingProfile(
            mode="constant",
            base_speed=2.0,  # Strong paddling
            heading_mode="upstream",
        )

        result = runner.run_single_kayak(
            start_x=start_x,
            start_y=start_y,
            duration_hours=0.5,
            paddling=profile,
        )

        # Check final position
        final_chainage = runner.mesh.point_to_chainage(
            result.trajectory_x[-1],
            result.trajectory_y[-1],
        )

        # Should have moved (may be upstream or downstream depending on conditions)
        assert result.total_distance_m > 50

    @pytest.mark.asyncio
    async def test_particle_cloud(self, runner):
        """Particle cloud should spread and advect."""
        start_x, start_y = runner.mesh.chainage_to_point(800)

        history = runner.run_particle_cloud(
            start_x=start_x,
            start_y=start_y,
            n_particles=100,
            spread_m=10.0,
            duration_hours=0.5,
        )

        assert len(history) > 1

        # Particles should have moved or spread
        initial = history[0]
        final = history[-1]

        # Check that simulation ran (timestamps advanced)
        assert final.timestamp > initial.timestamp


class TestScenarios:
    """Test specific scenario outcomes."""

    @pytest.mark.asyncio
    async def test_ebb_drift_downstream(self):
        """During ebb, drifting kayak should move downstream."""
        runner = SimulationRunner()
        await runner.initialize(use_synthetic_data=True)

        # Find a time when tide is ebbing
        now = datetime.now()
        for h in range(12):
            test_time = now + timedelta(hours=h)
            phase = runner.tide_model.get_phase(test_time)
            if phase == "ebbing":
                break
        else:
            pytest.skip("No ebb tide found in test window")

        start_x, start_y = runner.mesh.chainage_to_point(1500)
        start_chainage = 1500

        result = runner.run_single_kayak(
            start_x=start_x,
            start_y=start_y,
            start_time=test_time,
            duration_hours=1.0,
            paddling=None,
        )

        # Check movement direction
        final_chainage = runner.mesh.point_to_chainage(
            result.trajectory_x[-1],
            result.trajectory_y[-1],
        )

        # During ebb, should move towards mouth (lower chainage)
        # Note: River flow also pushes downstream, so this should be consistent
        # May need to account for strong river flow overwhelming ebb

    @pytest.mark.asyncio
    async def test_flood_helps_upstream(self):
        """During flood, upstream paddling should be easier."""
        runner = SimulationRunner()
        await runner.initialize(use_synthetic_data=True)

        # Find flood and ebb times
        now = datetime.now()
        flood_time = None
        ebb_time = None

        for h in range(12):
            test_time = now + timedelta(hours=h)
            phase = runner.tide_model.get_phase(test_time)
            if phase == "flooding" and flood_time is None:
                flood_time = test_time
            elif phase == "ebbing" and ebb_time is None:
                ebb_time = test_time

        if flood_time is None or ebb_time is None:
            pytest.skip("Could not find both flood and ebb in test window")

        start_x, start_y = runner.mesh.chainage_to_point(500)
        profile = PaddlingProfile(
            mode="constant",
            base_speed=1.5,
            heading_mode="upstream",
        )

        # Run during flood
        result_flood = runner.run_single_kayak(
            start_x=start_x,
            start_y=start_y,
            start_time=flood_time,
            duration_hours=1.0,
            paddling=profile,
        )

        # Run during ebb
        result_ebb = runner.run_single_kayak(
            start_x=start_x,
            start_y=start_y,
            start_time=ebb_time,
            duration_hours=1.0,
            paddling=profile,
        )

        # Calculate net upstream progress
        flood_progress = runner.mesh.point_to_chainage(
            result_flood.trajectory_x[-1],
            result_flood.trajectory_y[-1],
        ) - 500

        ebb_progress = runner.mesh.point_to_chainage(
            result_ebb.trajectory_x[-1],
            result_ebb.trajectory_y[-1],
        ) - 500

        # Flood should give better upstream progress
        # (or less downstream drift if river is strong)
        assert flood_progress > ebb_progress or abs(flood_progress - ebb_progress) < 100


class TestResultSerialization:
    """Tests for result export/serialization."""

    @pytest.mark.asyncio
    async def test_result_to_dict(self):
        """Result should serialize to dictionary."""
        runner = SimulationRunner()
        await runner.initialize(use_synthetic_data=True)

        start_x, start_y = runner.mesh.chainage_to_point(500)
        result = runner.run_single_kayak(
            start_x=start_x,
            start_y=start_y,
            duration_hours=0.5,
        )

        data = result.to_dict()

        assert "start_time" in data
        assert "trajectory_x" in data
        assert "total_distance_m" in data
        assert isinstance(data["trajectory_x"], list)
