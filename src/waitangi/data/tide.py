"""Tide prediction data from NIWA and harmonic calculations."""

from datetime import datetime, timedelta

import httpx
import numpy as np
from pydantic import BaseModel

from waitangi.core.config import DataSourceSettings, WaitangiLocation, get_settings
from waitangi.core.constants import (
    OMEGA_K1,
    OMEGA_M2,
    OMEGA_N2,
    OMEGA_O1,
    OMEGA_S2,
)


class TidePrediction(BaseModel):
    """Tide prediction at a specific time."""

    timestamp: datetime
    height_m: float  # Height above chart datum
    is_high: bool | None = None  # True=high, False=low, None=intermediate
    source: str = "calculated"  # "niwa", "linz", "calculated"


class TideHarmonics(BaseModel):
    """Harmonic constituent amplitudes and phases for a location.

    Reference: Waitangi River mouth, derived from Opua standard port.
    """

    # Amplitudes (meters)
    m2_amp: float = 0.95  # Principal lunar semidiurnal
    s2_amp: float = 0.18  # Principal solar semidiurnal
    n2_amp: float = 0.20  # Larger lunar elliptic
    k1_amp: float = 0.08  # Lunar diurnal
    o1_amp: float = 0.06  # Lunar diurnal

    # Phases (radians, relative to NZ standard time)
    m2_phase: float = 2.8
    s2_phase: float = 3.1
    n2_phase: float = 2.5
    k1_phase: float = 0.5
    o1_phase: float = 0.3

    # Mean sea level above chart datum (meters)
    z0: float = 1.45

    def calculate_height(self, t_hours: float) -> float:
        """Calculate tide height at time t (hours from reference epoch).

        Args:
            t_hours: Hours since reference epoch (J2000.0 or similar).

        Returns:
            Tide height in meters above chart datum.
        """
        height = self.z0
        height += self.m2_amp * np.cos(OMEGA_M2 * t_hours - self.m2_phase)
        height += self.s2_amp * np.cos(OMEGA_S2 * t_hours - self.s2_phase)
        height += self.n2_amp * np.cos(OMEGA_N2 * t_hours - self.n2_phase)
        height += self.k1_amp * np.cos(OMEGA_K1 * t_hours - self.k1_phase)
        height += self.o1_amp * np.cos(OMEGA_O1 * t_hours - self.o1_phase)
        return float(height)

    def calculate_velocity(self, t_hours: float) -> float:
        """Calculate tidal velocity (dh/dt) at time t.

        Velocity is proportional to dh/dt for a simple tidal prism model.

        Args:
            t_hours: Hours since reference epoch.

        Returns:
            Tidal velocity contribution in m/s (positive = flood, negative = ebb).
        """
        # dh/dt in m/hour
        dhdt = 0.0
        dhdt -= self.m2_amp * OMEGA_M2 * np.sin(OMEGA_M2 * t_hours - self.m2_phase)
        dhdt -= self.s2_amp * OMEGA_S2 * np.sin(OMEGA_S2 * t_hours - self.s2_phase)
        dhdt -= self.n2_amp * OMEGA_N2 * np.sin(OMEGA_N2 * t_hours - self.n2_phase)
        dhdt -= self.k1_amp * OMEGA_K1 * np.sin(OMEGA_K1 * t_hours - self.k1_phase)
        dhdt -= self.o1_amp * OMEGA_O1 * np.sin(OMEGA_O1 * t_hours - self.o1_phase)

        # Convert to m/s and scale to typical estuary current
        # Typical factor: 1 m tidal range → ~0.5-1 m/s max current in estuary
        velocity_scale = 0.8  # m/s per (m/hour) - calibration parameter
        return float(dhdt * velocity_scale / 3600)


# Reference epoch for harmonic calculations: 2020-01-01 00:00 NZST
TIDE_EPOCH = datetime(2020, 1, 1, 0, 0, 0)


def datetime_to_tide_hours(dt: datetime) -> float:
    """Convert datetime to hours since tide reference epoch."""
    delta = dt - TIDE_EPOCH
    return delta.total_seconds() / 3600


def calculate_tide_predictions(
    start_time: datetime,
    end_time: datetime,
    interval_minutes: int = 15,
    harmonics: TideHarmonics | None = None,
) -> list[TidePrediction]:
    """Calculate tide predictions using harmonic constituents.

    Args:
        start_time: Start of prediction period.
        end_time: End of prediction period.
        interval_minutes: Time step between predictions.
        harmonics: Harmonic constants. Defaults to Waitangi mouth.

    Returns:
        List of TidePrediction objects.
    """
    if harmonics is None:
        harmonics = TideHarmonics()

    predictions = []
    current_time = start_time
    dt = timedelta(minutes=interval_minutes)

    prev_height = None
    prev_prev_height = None

    while current_time <= end_time:
        t_hours = datetime_to_tide_hours(current_time)
        height = harmonics.calculate_height(t_hours)

        # Detect high/low
        is_high = None
        if prev_height is not None and prev_prev_height is not None:
            if prev_height > height and prev_height > prev_prev_height:
                # Previous point was a high
                predictions[-1].is_high = True
            elif prev_height < height and prev_height < prev_prev_height:
                # Previous point was a low
                predictions[-1].is_high = False

        predictions.append(
            TidePrediction(
                timestamp=current_time,
                height_m=round(height, 3),
                is_high=is_high,
                source="calculated",
            )
        )

        prev_prev_height = prev_height
        prev_height = height
        current_time += dt

    return predictions


async def fetch_tide_predictions(
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    hours_ahead: int = 168,  # 7 days
    port: str = "Opua",
    settings: DataSourceSettings | None = None,
) -> list[TidePrediction]:
    """Fetch tide predictions from NIWA or calculate from harmonics.

    Args:
        start_time: Start of prediction period. Defaults to now.
        end_time: End of prediction period.
        hours_ahead: Hours to predict if end_time not specified.
        port: Reference port name.
        settings: Data source settings.

    Returns:
        List of TidePrediction objects.
    """
    if settings is None:
        settings = get_settings().data_sources

    if start_time is None:
        start_time = datetime.now()
    if end_time is None:
        end_time = start_time + timedelta(hours=hours_ahead)

    # Try NIWA tide API first
    try:
        predictions = await _fetch_niwa_tides(start_time, end_time, port, settings)
        if predictions:
            return predictions
    except httpx.HTTPError:
        pass

    # Fall back to harmonic calculation
    return calculate_tide_predictions(start_time, end_time)


async def _fetch_niwa_tides(
    start_time: datetime,
    end_time: datetime,
    port: str,
    settings: DataSourceSettings,
) -> list[TidePrediction]:
    """Fetch tide predictions from NIWA tide service.

    Note: NIWA provides free tide predictions but may have access restrictions.
    """
    # NIWA tide service endpoint
    params = {
        "lat": WaitangiLocation.TIDE_PORT_LAT,
        "long": WaitangiLocation.TIDE_PORT_LON,
        "numberOfDays": int((end_time - start_time).days) + 1,
        "startDate": start_time.strftime("%Y-%m-%d"),
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{settings.niwa_base_url}/api/tides",
            params=params,
        )
        response.raise_for_status()

        data = response.json()
        predictions = []

        # Parse NIWA response format
        for item in data.get("values", []):
            predictions.append(
                TidePrediction(
                    timestamp=datetime.fromisoformat(item["time"]),
                    height_m=item["value"],
                    is_high=item.get("type") == "HIGH",
                    source="niwa",
                )
            )

        return predictions


def get_current_tide_state(harmonics: TideHarmonics | None = None) -> dict:
    """Get current tide height, velocity, and phase.

    Returns:
        Dictionary with current tide state information.
    """
    if harmonics is None:
        harmonics = TideHarmonics()

    now = datetime.now()
    t_hours = datetime_to_tide_hours(now)

    height = harmonics.calculate_height(t_hours)
    velocity = harmonics.calculate_velocity(t_hours)

    # Determine tide phase
    if velocity > 0.05:
        phase = "flooding"
    elif velocity < -0.05:
        phase = "ebbing"
    else:
        # Near slack water - check if high or low
        t_plus = datetime_to_tide_hours(now + timedelta(minutes=30))
        height_plus = harmonics.calculate_height(t_plus)
        if height_plus > height:
            phase = "slack_low"
        else:
            phase = "slack_high"

    return {
        "timestamp": now,
        "height_m": round(height, 3),
        "velocity_ms": round(velocity, 4),
        "phase": phase,
        "hours_to_next_high": _hours_to_next_extreme(t_hours, harmonics, find_high=True),
        "hours_to_next_low": _hours_to_next_extreme(t_hours, harmonics, find_high=False),
    }


def _hours_to_next_extreme(
    t_hours: float, harmonics: TideHarmonics, find_high: bool
) -> float:
    """Calculate hours until next high or low tide."""
    dt = 0.25  # 15-minute steps
    prev_height = harmonics.calculate_height(t_hours)

    for i in range(1, 100):  # Max 25 hours ahead
        t = t_hours + i * dt
        height = harmonics.calculate_height(t)

        if find_high:
            if height < prev_height:
                # Just passed a high
                return (i - 1) * dt
        else:
            if height > prev_height:
                # Just passed a low
                return (i - 1) * dt

        prev_height = height

    return 12.5  # Fallback to half tidal cycle
