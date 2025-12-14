"""Rainfall data ingestion for catchment runoff modeling."""

from datetime import datetime, timedelta

import httpx
import numpy as np
from pydantic import BaseModel

from waitangi.core.config import DataSourceSettings, get_settings


class RainfallReading(BaseModel):
    """Single rainfall observation or forecast."""

    timestamp: datetime
    rainfall_mm: float
    is_forecast: bool = False
    confidence: float | None = None  # 0-1 for forecasts


class RainfallData(BaseModel):
    """Time series of rainfall data (observations + forecast)."""

    site_name: str
    readings: list[RainfallReading]
    last_updated: datetime

    @property
    def observations(self) -> list[RainfallReading]:
        """Get only observed (historical) readings."""
        return [r for r in self.readings if not r.is_forecast]

    @property
    def forecasts(self) -> list[RainfallReading]:
        """Get only forecast readings."""
        return [r for r in self.readings if r.is_forecast]

    @property
    def timestamps(self) -> np.ndarray:
        """Get timestamps as numpy datetime64 array."""
        return np.array([r.timestamp for r in self.readings], dtype="datetime64[s]")

    @property
    def rainfall(self) -> np.ndarray:
        """Get rainfall values as numpy array."""
        return np.array([r.rainfall_mm for r in self.readings], dtype=np.float64)

    def total_past_hours(self, hours: int) -> float:
        """Calculate total rainfall in the past N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return sum(r.rainfall_mm for r in self.observations if r.timestamp >= cutoff)

    def total_forecast_hours(self, hours: int) -> float:
        """Calculate total forecast rainfall in the next N hours."""
        now = datetime.now()
        cutoff = now + timedelta(hours=hours)
        return sum(
            r.rainfall_mm for r in self.forecasts if now <= r.timestamp <= cutoff
        )


async def fetch_rainfall_data(
    site_name: str = "Waitangi Forest",
    hours_back: int = 72,
    hours_ahead: int = 48,
    settings: DataSourceSettings | None = None,
) -> RainfallData:
    """Fetch rainfall observations and forecasts.

    Args:
        site_name: Name of the rainfall monitoring site.
        hours_back: Hours of historical data to retrieve.
        hours_ahead: Hours of forecast data to retrieve.
        settings: Data source settings.

    Returns:
        RainfallData with observations and forecasts.

    Note:
        This combines NRC historical data with MetService forecasts.
    """
    if settings is None:
        settings = get_settings().data_sources

    end_obs = datetime.now()
    start_obs = end_obs - timedelta(hours=hours_back)
    end_forecast = end_obs + timedelta(hours=hours_ahead)

    readings: list[RainfallReading] = []

    # Try to fetch historical observations from NRC
    try:
        obs_readings = await _fetch_nrc_rainfall(
            site_name, start_obs, end_obs, settings
        )
        readings.extend(obs_readings)
    except httpx.HTTPError:
        # Fall back to synthetic data
        readings.extend(_generate_synthetic_rainfall(start_obs, end_obs, is_forecast=False))

    # Try to fetch forecast from MetService
    if settings.metservice_api_key:
        try:
            forecast_readings = await _fetch_metservice_rainfall(
                end_obs, end_forecast, settings
            )
            readings.extend(forecast_readings)
        except httpx.HTTPError:
            readings.extend(
                _generate_synthetic_rainfall(end_obs, end_forecast, is_forecast=True)
            )
    else:
        # No API key - use synthetic forecast
        readings.extend(
            _generate_synthetic_rainfall(end_obs, end_forecast, is_forecast=True)
        )

    # Sort by timestamp
    readings.sort(key=lambda r: r.timestamp)

    return RainfallData(
        site_name=site_name,
        readings=readings,
        last_updated=datetime.now(),
    )


async def _fetch_nrc_rainfall(
    site_name: str,
    start_time: datetime,
    end_time: datetime,
    settings: DataSourceSettings,
) -> list[RainfallReading]:
    """Fetch rainfall observations from NRC Hilltop Server."""
    params = {
        "Service": "Hilltop",
        "Request": "GetData",
        "Site": site_name,
        "Measurement": "Rainfall",
        "From": start_time.strftime("%Y-%m-%d %H:%M"),
        "To": end_time.strftime("%Y-%m-%d %H:%M"),
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{settings.nrc_base_url}/data.hts",
            params=params,
        )
        response.raise_for_status()

        # Parse response (placeholder - actual parsing needed)
        return []


async def _fetch_metservice_rainfall(
    start_time: datetime,
    end_time: datetime,
    settings: DataSourceSettings,
) -> list[RainfallReading]:
    """Fetch rainfall forecast from MetService API."""
    # MetService Point Forecast API
    # Note: Requires enterprise subscription
    headers = {"x-api-key": settings.metservice_api_key}

    # Bay of Islands location
    params = {
        "lat": -35.27,
        "lon": 174.09,
        "hours": int((end_time - start_time).total_seconds() / 3600),
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{settings.metservice_base_url}/pointforecast/rain",
            params=params,
            headers=headers,
        )
        response.raise_for_status()

        # Parse response (placeholder - actual parsing needed)
        return []


def _generate_synthetic_rainfall(
    start_time: datetime,
    end_time: datetime,
    is_forecast: bool = False,
    interval_hours: int = 1,
) -> list[RainfallReading]:
    """Generate synthetic rainfall data for development.

    Creates realistic rainfall patterns for NZ coastal catchment:
    - Frequent light rain
    - Occasional heavier events
    - Higher probability during certain hours
    """
    readings = []
    current_time = start_time
    dt_hours = timedelta(hours=interval_hours)

    rng = np.random.default_rng(123 if not is_forecast else 456)

    # Event state machine
    in_event = False
    event_hours_remaining = 0
    event_intensity = 0.0

    while current_time <= end_time:
        rainfall = 0.0

        if in_event:
            # During a rain event
            event_hours_remaining -= 1
            # Intensity varies within event
            rainfall = event_intensity * (0.5 + rng.random())
            if event_hours_remaining <= 0:
                in_event = False
        else:
            # Probability of starting a new event
            # Higher in early morning and evening (typical NZ pattern)
            hour = current_time.hour
            base_prob = 0.05
            if 4 <= hour <= 8 or 16 <= hour <= 20:
                base_prob = 0.12

            if rng.random() < base_prob:
                in_event = True
                event_hours_remaining = int(rng.exponential(4)) + 1
                # Event intensity (mm/hour)
                event_intensity = rng.exponential(2.0)
                rainfall = event_intensity * (0.5 + rng.random())

        # Add small chance of isolated shower even outside events
        if not in_event and rng.random() < 0.02:
            rainfall = rng.exponential(0.5)

        readings.append(
            RainfallReading(
                timestamp=current_time,
                rainfall_mm=round(max(0, rainfall), 2),
                is_forecast=is_forecast,
                confidence=0.8 - 0.3 * ((current_time - start_time).days / 7)
                if is_forecast
                else None,
            )
        )

        current_time += dt_hours

    return readings
