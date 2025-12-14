"""Tests for river discharge model."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from waitangi.data.gauge import GaugeData, GaugeReading, _generate_synthetic_gauge_data
from waitangi.data.rainfall import RainfallData, RainfallReading
from waitangi.models.river import CatchmentState, RiverDischargeModel


class TestCatchmentState:
    """Tests for catchment wetness state."""

    def test_initial_state(self):
        """Should have reasonable default values."""
        state = CatchmentState()
        assert 0 <= state.wetness <= 1
        assert state.soil_moisture_deficit >= 0

    def test_update_from_rainfall(self):
        """Rainfall should increase wetness."""
        state = CatchmentState()
        initial_wetness = state.wetness

        state.update_from_rainfall(30.0)  # 30mm in 24h

        assert state.wetness > initial_wetness
        assert state.recent_rainfall_mm == 30.0
        assert state.last_updated is not None

    def test_saturation_limit(self):
        """Wetness should cap at 1.0."""
        state = CatchmentState()
        state.update_from_rainfall(100.0)  # Heavy rain

        assert state.wetness <= 1.0


class TestRiverDischargeModel:
    """Tests for river discharge calculations."""

    def test_create_default(self):
        """Should create model with default settings."""
        model = RiverDischargeModel.create()
        assert model.settings.q_base > 0
        assert model.catchment_state is not None

    def test_baseflow_without_data(self):
        """Without data, should return baseflow."""
        model = RiverDischargeModel.create()
        q = model.get_discharge(datetime.now())

        # Should be close to baseflow
        assert q > 0
        assert abs(q - model.settings.q_base) < 1.0

    def test_ingest_gauge_data(self):
        """Should accept gauge data and interpolate."""
        model = RiverDischargeModel.create()

        # Create synthetic gauge data
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        readings = _generate_synthetic_gauge_data(start_time, end_time)

        gauge_data = GaugeData(
            site_id="test",
            site_name="Test Site",
            readings=readings,
            last_updated=datetime.now(),
        )

        model.ingest_gauge_data(gauge_data)

        # Now discharge should be interpolated from gauge
        q = model.get_discharge(end_time - timedelta(hours=1))
        assert q > 0

    def test_rainfall_response(self):
        """Discharge should increase after rainfall."""
        model = RiverDischargeModel.create()

        # Create rainfall data with a pulse
        now = datetime.now()
        readings = [
            RainfallReading(timestamp=now - timedelta(hours=h), rainfall_mm=0.0)
            for h in range(48, 6, -1)
        ]
        # Add rainfall pulse 6-3 hours ago
        readings.extend([
            RainfallReading(timestamp=now - timedelta(hours=h), rainfall_mm=10.0)
            for h in range(6, 3, -1)
        ])
        readings.extend([
            RainfallReading(timestamp=now - timedelta(hours=h), rainfall_mm=0.0)
            for h in range(3, 0, -1)
        ])

        rainfall_data = RainfallData(
            site_name="Test",
            readings=readings,
            last_updated=now,
        )

        model.ingest_rainfall_data(rainfall_data)

        # Current discharge should be higher than baseflow
        q_now = model.get_discharge(now)
        assert q_now >= model.settings.q_base

    def test_velocity_at_mouth(self):
        """Should convert discharge to velocity."""
        model = RiverDischargeModel.create()
        v = model.get_velocity_at_mouth(datetime.now())

        # Velocity should be positive and reasonable
        assert v > 0
        assert v < 5  # < 5 m/s is reasonable

    def test_velocity_field_decay(self):
        """Velocity should decay upstream."""
        model = RiverDischargeModel.create()
        now = datetime.now()

        v_mouth = model.get_velocity_field(now, 0.0)
        v_mid = model.get_velocity_field(now, 1500.0)
        v_upstream = model.get_velocity_field(now, 3000.0)

        # Should decay exponentially
        assert float(v_mouth) > float(v_mid) > float(v_upstream)

    def test_discharge_series(self):
        """Should generate time series of discharge."""
        model = RiverDischargeModel.create()
        start = datetime.now()
        end = start + timedelta(hours=6)

        times, discharges = model.get_discharge_series(start, end)

        assert len(times) == len(discharges)
        assert len(times) > 10
        assert all(d > 0 for d in discharges)
