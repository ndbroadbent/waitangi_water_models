"""Tests for tide model."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from waitangi.data.tide import (
    TideHarmonics,
    calculate_tide_predictions,
    datetime_to_tide_hours,
)
from waitangi.models.tide import TideModel


class TestTideHarmonics:
    """Tests for harmonic tide calculations."""

    def test_mean_height(self):
        """Mean tide should return mean sea level."""
        harmonics = TideHarmonics()
        # Average over full tidal cycle
        heights = [harmonics.calculate_height(t) for t in np.linspace(0, 25, 1000)]
        mean = np.mean(heights)
        assert abs(mean - harmonics.z0) < 0.05

    def test_semidiurnal_period(self):
        """Tide should have ~12.4 hour period."""
        harmonics = TideHarmonics()

        # Find first high after t=0
        heights = [harmonics.calculate_height(t) for t in np.linspace(0, 15, 1000)]
        high_idx = np.argmax(heights)
        first_high_t = high_idx * 15 / 1000

        # Find second high
        heights2 = [harmonics.calculate_height(t) for t in np.linspace(14, 28, 1000)]
        high_idx2 = np.argmax(heights2)
        second_high_t = 14 + high_idx2 * 14 / 1000

        period = second_high_t - first_high_t
        # Should be close to M2 period (12.42 hours)
        assert 12.0 < period < 13.0

    def test_spring_neap_amplitude_variation(self):
        """Amplitude should vary over ~14 day cycle."""
        harmonics = TideHarmonics()

        # Calculate daily ranges over 30 days
        daily_ranges = []
        for day in range(30):
            t_start = day * 24
            heights = [harmonics.calculate_height(t_start + h) for h in range(25)]
            daily_range = max(heights) - min(heights)
            daily_ranges.append(daily_range)

        # Should have variation (spring/neap)
        assert max(daily_ranges) > min(daily_ranges) * 1.1

    def test_velocity_sign_change(self):
        """Velocity should change sign at high/low tide."""
        harmonics = TideHarmonics()

        # Find a high tide
        heights = [harmonics.calculate_height(t) for t in np.linspace(0, 13, 1000)]
        high_idx = np.argmax(heights)
        high_t = high_idx * 13 / 1000

        # Velocity should be near zero at high tide
        v_at_high = harmonics.calculate_velocity(high_t)
        assert abs(v_at_high) < 0.05

        # Velocity before high should be positive (flooding)
        v_before = harmonics.calculate_velocity(high_t - 2)
        assert v_before > 0

        # Velocity after high should be negative (ebbing)
        v_after = harmonics.calculate_velocity(high_t + 2)
        assert v_after < 0


class TestTideModel:
    """Tests for TideModel integration."""

    def test_create_default(self):
        """Should create model with default settings."""
        model = TideModel.create()
        assert model.harmonics.z0 > 0
        assert model.harmonics.m2_amp > 0

    def test_height_range(self):
        """Heights should be within reasonable range."""
        model = TideModel.create()
        now = datetime.now()

        heights = [
            model.get_height(now + timedelta(hours=h))
            for h in range(25)
        ]

        # Should be between 0 and 4 meters typically
        assert min(heights) > 0
        assert max(heights) < 4

    def test_phase_detection(self):
        """Should correctly detect tide phase."""
        model = TideModel.create()
        now = datetime.now()

        # Check multiple times to find different phases
        phases = set()
        for h in range(24):
            phase = model.get_phase(now + timedelta(hours=h))
            phases.add(phase)

        # Should detect multiple different phases over 24h
        # Could be flooding, ebbing, slack_high, or slack_low
        assert len(phases) >= 2

    def test_velocity_field_decay(self):
        """Tidal velocity should decay upstream."""
        model = TideModel.create()
        now = datetime.now()

        v_mouth = model.get_velocity_field(now, 0.0)
        v_mid = model.get_velocity_field(now, 1500.0)
        v_upstream = model.get_velocity_field(now, 3000.0)

        # Velocity should decrease upstream
        assert abs(float(v_mouth)) >= abs(float(v_mid))
        assert abs(float(v_mid)) >= abs(float(v_upstream))


class TestTidePredictions:
    """Tests for tide prediction generation."""

    def test_calculate_predictions(self):
        """Should generate predictions over specified period."""
        start = datetime.now()
        end = start + timedelta(days=1)

        predictions = calculate_tide_predictions(start, end)

        assert len(predictions) > 0
        assert predictions[0].timestamp >= start
        assert predictions[-1].timestamp <= end

    def test_predictions_have_extremes(self):
        """Should identify high and low tides."""
        start = datetime.now()
        end = start + timedelta(days=1)

        predictions = calculate_tide_predictions(start, end)

        # Should have some highs and lows marked
        highs = [p for p in predictions if p.is_high is True]
        lows = [p for p in predictions if p.is_high is False]

        # Expect ~2 highs and 2 lows per day
        assert len(highs) >= 1
        assert len(lows) >= 1
