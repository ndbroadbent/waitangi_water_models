"""River discharge model with rainfall-runoff response.

Combines:
1. Live gauge data for nowcasting (0-12h)
2. Rainfall-driven forecast for longer horizons
3. Catchment wetness state for varying runoff response
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import jax.numpy as jnp
import numpy as np
from jax import Array

from waitangi.core.config import RiverSettings, get_settings
from waitangi.data.gauge import GaugeData
from waitangi.data.rainfall import RainfallData


@dataclass
class CatchmentState:
    """State of the catchment for runoff modeling."""

    # Wetness index (0=dry, 1=saturated)
    wetness: float = 0.3

    # Soil moisture deficit (mm)
    soil_moisture_deficit: float = 20.0

    # Recent rainfall total (mm in last 24h)
    recent_rainfall_mm: float = 0.0

    # Last update time
    last_updated: datetime | None = None

    def update_from_rainfall(self, rainfall_mm_24h: float) -> None:
        """Update catchment state from recent rainfall."""
        # Simple empirical relationship
        # Wetness increases with rainfall, decays over time
        self.recent_rainfall_mm = rainfall_mm_24h

        # Soil moisture deficit decreases with rain
        self.soil_moisture_deficit = max(0, 30 - rainfall_mm_24h)

        # Wetness index based on recent rain and deficit
        self.wetness = min(1.0, rainfall_mm_24h / 50 + 0.2)

        self.last_updated = datetime.now()


@dataclass
class RiverDischargeModel:
    """Model for river discharge Q(t) from gauge and rainfall data.

    The model uses a unit hydrograph approach:
    Q(t) = Q_base + Q_rain(t)

    Where Q_rain is the convolution of rainfall with an exponential
    unit hydrograph:
    Q_rain(t) = Σ [ R(t_i) * K * exp(-(t - t_i - delay)/τ) * step(t - t_i - delay) ]
    """

    settings: RiverSettings
    catchment_state: CatchmentState

    # Cached rainfall for convolution
    _rainfall_times: Array | None = None
    _rainfall_values: Array | None = None

    # Cached gauge data for nowcasting
    _gauge_times: Array | None = None
    _gauge_flows: Array | None = None

    @classmethod
    def create(cls, settings: RiverSettings | None = None) -> "RiverDischargeModel":
        """Create a new discharge model with default state."""
        if settings is None:
            settings = get_settings().river
        return cls(settings=settings, catchment_state=CatchmentState())

    def ingest_gauge_data(self, gauge_data: GaugeData) -> None:
        """Ingest gauge data for nowcasting.

        Args:
            gauge_data: River gauge observations.
        """
        flows = gauge_data.flows
        valid_mask = ~np.isnan(flows)

        if not np.any(valid_mask):
            return

        timestamps = gauge_data.timestamps[valid_mask]
        flows = flows[valid_mask]

        # Convert to seconds since epoch for interpolation
        epoch = np.datetime64(0, "s")
        times_seconds = (timestamps - epoch).astype(np.float64)

        self._gauge_times = jnp.asarray(times_seconds)
        self._gauge_flows = jnp.asarray(flows)

    def ingest_rainfall_data(self, rainfall_data: RainfallData) -> None:
        """Ingest rainfall data for runoff modeling.

        Args:
            rainfall_data: Rainfall observations and forecasts.
        """
        rainfall = rainfall_data.rainfall
        timestamps = rainfall_data.timestamps

        # Convert to seconds since epoch
        epoch = np.datetime64(0, "s")
        times_seconds = (timestamps - epoch).astype(np.float64)

        self._rainfall_times = jnp.asarray(times_seconds)
        self._rainfall_values = jnp.asarray(rainfall)

        # Update catchment state
        self.catchment_state.update_from_rainfall(rainfall_data.total_past_hours(24))

    def get_discharge(self, timestamp: datetime) -> float:
        """Get river discharge at a given time.

        For nowcasting (recent times), uses gauge data directly.
        For forecasting, uses rainfall-runoff model.

        Args:
            timestamp: Time for discharge calculation.

        Returns:
            Discharge in m³/s.
        """
        # Convert timestamp to seconds since epoch
        epoch = datetime(1970, 1, 1)
        t_seconds = (timestamp - epoch).total_seconds()

        # Try gauge data first (for nowcasting)
        if self._gauge_times is not None and self._gauge_flows is not None:
            latest_gauge_time = float(self._gauge_times[-1])

            # If within gauge data range, interpolate
            if t_seconds <= latest_gauge_time + 3600:  # 1 hour buffer
                return self._interpolate_gauge(t_seconds)

        # Otherwise use rainfall-runoff model
        return self._calculate_from_rainfall(t_seconds)

    def get_discharge_series(
        self,
        start_time: datetime,
        end_time: datetime,
        interval_seconds: float = 900,
    ) -> tuple[Array, Array]:
        """Get discharge time series.

        Args:
            start_time: Start of series.
            end_time: End of series.
            interval_seconds: Time step.

        Returns:
            Tuple of (timestamps, discharges) as JAX arrays.
        """
        epoch = datetime(1970, 1, 1)
        t_start = (start_time - epoch).total_seconds()
        t_end = (end_time - epoch).total_seconds()

        times = jnp.arange(t_start, t_end, interval_seconds)
        discharges = jnp.array([
            self.get_discharge(epoch + timedelta(seconds=float(t)))
            for t in times
        ])

        return times, discharges

    def get_velocity_at_mouth(self, timestamp: datetime) -> float:
        """Convert discharge to surface velocity at river mouth.

        Uses Q = A * V relationship with configured cross-section.

        Args:
            timestamp: Time for calculation.

        Returns:
            Surface velocity in m/s (positive = downstream/seaward).
        """
        discharge = self.get_discharge(timestamp)
        cross_section = self.settings.cross_section_m2

        # Surface velocity is typically 1.2-1.5x mean velocity
        mean_velocity = discharge / cross_section
        surface_velocity = mean_velocity * 1.3

        return surface_velocity

    def get_velocity_field(
        self, timestamp: datetime, chainage_m: float | Array
    ) -> float | Array:
        """Get river velocity at distance from mouth.

        Velocity decays exponentially upstream:
        v(x) = V0 * exp(-x / L)

        Args:
            timestamp: Time for calculation.
            chainage_m: Distance(s) from mouth in meters.

        Returns:
            Velocity in m/s (positive = downstream).
        """
        v0 = self.get_velocity_at_mouth(timestamp)
        decay_length = self.settings.decay_length_m

        chainage = jnp.asarray(chainage_m)
        return v0 * jnp.exp(-chainage / decay_length)

    def _interpolate_gauge(self, t_seconds: float) -> float:
        """Interpolate gauge data at given time."""
        if self._gauge_times is None or self._gauge_flows is None:
            return self.settings.q_base

        # Clamp to data range
        t_clamped = jnp.clip(
            t_seconds, float(self._gauge_times[0]), float(self._gauge_times[-1])
        )

        # Linear interpolation
        return float(jnp.interp(t_clamped, self._gauge_times, self._gauge_flows))

    def _calculate_from_rainfall(self, t_seconds: float) -> float:
        """Calculate discharge from rainfall using unit hydrograph."""
        q_base = self.settings.q_base

        if self._rainfall_times is None or self._rainfall_values is None:
            return q_base

        # Effective runoff coefficient (varies with wetness)
        k_eff = self.settings.runoff_coefficient * (
            0.5 + 0.5 * self.catchment_state.wetness
        )

        # Convert time constants to seconds
        tau_s = self.settings.tau_hours * 3600
        delay_s = self.settings.delay_hours * 3600

        # Catchment area in m²
        area_m2 = self.settings.catchment_area_km2 * 1e6

        # Convolution with exponential unit hydrograph
        q_rain = 0.0

        for i in range(len(self._rainfall_times)):
            t_rain = float(self._rainfall_times[i])
            r_mm = float(self._rainfall_values[i])

            if r_mm <= 0:
                continue

            # Time since rainfall (accounting for delay)
            dt = t_seconds - t_rain - delay_s

            if dt > 0:
                # Exponential decay response
                # Q = R * K * A * exp(-t/tau) / tau
                # Units: mm/hr * (-) * m² * (-) / hr → m³/s
                response = k_eff * (r_mm / 1000) * area_m2 * np.exp(-dt / tau_s) / tau_s
                q_rain += response

        return q_base + q_rain
