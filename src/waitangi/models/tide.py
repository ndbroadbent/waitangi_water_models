"""Tide model for water level and velocity at river mouth.

Supports both harmonic calculation and API-driven predictions.
"""

from dataclasses import dataclass
from datetime import datetime

import jax.numpy as jnp
from jax import Array

from waitangi.core.config import TideSettings, get_settings
from waitangi.data.tide import (
    TideHarmonics,
    TidePrediction,
    datetime_to_tide_hours,
)


@dataclass
class TideModel:
    """Model for tidal water level and velocity.

    The tide creates a time-varying boundary condition at the river mouth,
    driving flood (incoming) and ebb (outgoing) currents.
    """

    settings: TideSettings
    harmonics: TideHarmonics

    # Cached API predictions (if available)
    _predictions: list[TidePrediction] | None = None

    @classmethod
    def create(cls, settings: TideSettings | None = None) -> "TideModel":
        """Create a tide model with default harmonics."""
        if settings is None:
            settings = get_settings().tide

        harmonics = TideHarmonics(
            z0=settings.mean_sea_level,
            # Scale amplitudes based on spring/neap settings
            m2_amp=0.95 * (settings.spring_amplitude / 1.4),
            s2_amp=0.18 * (settings.spring_amplitude / 1.4),
            n2_amp=0.20 * (settings.spring_amplitude / 1.4),
        )

        return cls(settings=settings, harmonics=harmonics)

    def ingest_predictions(self, predictions: list[TidePrediction]) -> None:
        """Ingest tide predictions from API for improved accuracy.

        Predictions are used to refine the harmonic model.
        """
        self._predictions = predictions

        # If we have high/low tide points, calibrate harmonics
        extremes = [p for p in predictions if p.is_high is not None]
        if len(extremes) >= 4:
            self._calibrate_from_extremes(extremes)

    def get_height(self, timestamp: datetime) -> float:
        """Get tide height at given time.

        Args:
            timestamp: Time for height calculation.

        Returns:
            Tide height in meters above chart datum.
        """
        # Use API predictions if available and in range
        if self._predictions:
            height = self._interpolate_predictions(timestamp)
            if height is not None:
                return height

        # Fall back to harmonic calculation
        t_hours = datetime_to_tide_hours(timestamp)
        return self.harmonics.calculate_height(t_hours)

    def get_velocity(self, timestamp: datetime) -> float:
        """Get tidal current velocity at river mouth.

        Positive = flood (incoming), negative = ebb (outgoing).

        Args:
            timestamp: Time for velocity calculation.

        Returns:
            Tidal velocity in m/s.
        """
        t_hours = datetime_to_tide_hours(timestamp)
        return self.harmonics.calculate_velocity(t_hours)

    def get_velocity_field(
        self, timestamp: datetime, chainage_m: float | Array
    ) -> float | Array:
        """Get tidal velocity at distance from mouth.

        Tidal influence decreases upstream:
        v_tide(x) = v_mouth * f(x)

        where f(x) ramps from 1 at mouth to 0 upstream.

        Args:
            timestamp: Time for calculation.
            chainage_m: Distance(s) from mouth in meters.

        Returns:
            Tidal velocity in m/s.
        """
        v_mouth = self.get_velocity(timestamp)
        chainage = jnp.asarray(chainage_m)

        # Tidal penetration factor
        # Uses a sigmoidal transition centered at penetration_length
        penetration_length = 2000.0  # meters - calibration parameter
        transition_width = 500.0

        f_tide = 1.0 / (1.0 + jnp.exp((chainage - penetration_length) / transition_width))

        return v_mouth * f_tide

    def get_phase(self, timestamp: datetime) -> str:
        """Get current tidal phase.

        Returns:
            One of: "flooding", "ebbing", "slack_high", "slack_low"
        """
        velocity = self.get_velocity(timestamp)

        if velocity > 0.05:
            return "flooding"
        elif velocity < -0.05:
            return "ebbing"
        else:
            # Near slack - determine if high or low
            height_now = self.get_height(timestamp)
            height_soon = self.get_height(
                datetime.fromtimestamp(timestamp.timestamp() + 1800)
            )
            if height_soon > height_now:
                return "slack_low"
            else:
                return "slack_high"

    def get_height_series(
        self,
        start_time: datetime,
        end_time: datetime,
        interval_seconds: float = 900,
    ) -> tuple[Array, Array]:
        """Get tide height time series.

        Args:
            start_time: Start of series.
            end_time: End of series.
            interval_seconds: Time step.

        Returns:
            Tuple of (timestamps_seconds, heights_m) as JAX arrays.
        """
        from datetime import timedelta

        epoch = datetime(1970, 1, 1)
        t_start = (start_time - epoch).total_seconds()
        t_end = (end_time - epoch).total_seconds()

        times = jnp.arange(t_start, t_end, interval_seconds)
        heights = jnp.array([
            self.get_height(epoch + timedelta(seconds=float(t)))
            for t in times
        ])

        return times, heights

    def _interpolate_predictions(self, timestamp: datetime) -> float | None:
        """Interpolate API predictions at given time."""
        if not self._predictions:
            return None

        # Find bracketing predictions
        for i, pred in enumerate(self._predictions):
            if pred.timestamp >= timestamp:
                if i == 0:
                    return pred.height_m

                prev = self._predictions[i - 1]
                total_delta = (pred.timestamp - prev.timestamp).total_seconds()
                t_delta = (timestamp - prev.timestamp).total_seconds()
                alpha = t_delta / total_delta if total_delta > 0 else 0

                return prev.height_m + alpha * (pred.height_m - prev.height_m)

        return None

    def _calibrate_from_extremes(self, extremes: list[TidePrediction]) -> None:
        """Calibrate harmonic amplitudes from observed high/low tides.

        This adjusts the model to match recent observations while
        maintaining the correct timing from harmonic constituents.
        """
        highs = [p for p in extremes if p.is_high]
        lows = [p for p in extremes if not p.is_high]

        if not highs or not lows:
            return

        # Calculate mean high and low water
        mhw = sum(p.height_m for p in highs) / len(highs)
        mlw = sum(p.height_m for p in lows) / len(lows)

        # Update mean level and primary amplitude
        new_z0 = (mhw + mlw) / 2
        new_range = mhw - mlw
        amplitude_ratio = new_range / (2 * self.harmonics.m2_amp)

        self.harmonics.z0 = new_z0
        self.harmonics.m2_amp *= amplitude_ratio
        self.harmonics.s2_amp *= amplitude_ratio
        self.harmonics.n2_amp *= amplitude_ratio
