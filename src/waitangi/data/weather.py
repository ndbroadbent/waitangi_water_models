"""Weather data ingestion for wind modeling."""

from datetime import datetime, timedelta

import httpx
import numpy as np
from pydantic import BaseModel

from waitangi.core.config import DataSourceSettings, get_settings


class WeatherForecast(BaseModel):
    """Single weather forecast point."""

    timestamp: datetime
    wind_speed_ms: float  # m/s at reference height
    wind_direction_deg: float  # Degrees from north (meteorological convention)
    wind_gust_ms: float | None = None
    temperature_c: float | None = None
    pressure_hpa: float | None = None
    humidity_pct: float | None = None
    confidence: float | None = None  # 0-1

    @property
    def wind_direction_rad(self) -> float:
        """Wind direction in radians (mathematical convention, counterclockwise from east)."""
        # Convert from meteorological (from north, clockwise) to math convention
        return np.radians(90 - self.wind_direction_deg)

    @property
    def wind_u(self) -> float:
        """Eastward wind component (m/s)."""
        return self.wind_speed_ms * np.cos(self.wind_direction_rad)

    @property
    def wind_v(self) -> float:
        """Northward wind component (m/s)."""
        return self.wind_speed_ms * np.sin(self.wind_direction_rad)


class WeatherData(BaseModel):
    """Time series of weather forecasts."""

    site_name: str
    lat: float
    lon: float
    forecasts: list[WeatherForecast]
    last_updated: datetime

    @property
    def timestamps(self) -> np.ndarray:
        """Get timestamps as numpy datetime64 array."""
        return np.array([f.timestamp for f in self.forecasts], dtype="datetime64[s]")

    @property
    def wind_speeds(self) -> np.ndarray:
        """Get wind speeds as numpy array (m/s)."""
        return np.array([f.wind_speed_ms for f in self.forecasts], dtype=np.float64)

    @property
    def wind_directions(self) -> np.ndarray:
        """Get wind directions as numpy array (degrees)."""
        return np.array([f.wind_direction_deg for f in self.forecasts], dtype=np.float64)

    def interpolate_wind(self, timestamp: datetime) -> tuple[float, float]:
        """Interpolate wind speed and direction at a given time.

        Returns:
            Tuple of (wind_speed_ms, wind_direction_deg).
        """
        if not self.forecasts:
            return 0.0, 0.0

        # Find bracketing forecasts
        for i, f in enumerate(self.forecasts):
            if f.timestamp >= timestamp:
                if i == 0:
                    return f.wind_speed_ms, f.wind_direction_deg

                prev = self.forecasts[i - 1]
                total_delta = (f.timestamp - prev.timestamp).total_seconds()
                t_delta = (timestamp - prev.timestamp).total_seconds()
                alpha = t_delta / total_delta if total_delta > 0 else 0

                # Linear interpolation for speed
                speed = prev.wind_speed_ms + alpha * (f.wind_speed_ms - prev.wind_speed_ms)

                # Circular interpolation for direction
                d1 = prev.wind_direction_deg
                d2 = f.wind_direction_deg
                diff = (d2 - d1 + 180) % 360 - 180
                direction = (d1 + alpha * diff) % 360

                return speed, direction

        # Past end of forecast - use last value
        last = self.forecasts[-1]
        return last.wind_speed_ms, last.wind_direction_deg


async def fetch_weather_forecast(
    lat: float = -35.27,
    lon: float = 174.09,
    hours_ahead: int = 168,
    settings: DataSourceSettings | None = None,
) -> WeatherData:
    """Fetch weather forecast from MetService API.

    Args:
        lat: Latitude (defaults to Waitangi River area).
        lon: Longitude.
        hours_ahead: Hours of forecast to retrieve.
        settings: Data source settings.

    Returns:
        WeatherData with forecast time series.
    """
    if settings is None:
        settings = get_settings().data_sources

    forecasts: list[WeatherForecast] = []

    if settings.metservice_api_key:
        try:
            forecasts = await _fetch_metservice_weather(lat, lon, hours_ahead, settings)
        except httpx.HTTPError:
            pass

    # Fall back to synthetic data if no API or fetch failed
    if not forecasts:
        forecasts = _generate_synthetic_weather(hours_ahead)

    return WeatherData(
        site_name="Waitangi Area",
        lat=lat,
        lon=lon,
        forecasts=forecasts,
        last_updated=datetime.now(),
    )


async def _fetch_metservice_weather(
    lat: float,
    lon: float,
    hours_ahead: int,
    settings: DataSourceSettings,
) -> list[WeatherForecast]:
    """Fetch weather from MetService Point Forecast API."""
    headers = {"x-api-key": settings.metservice_api_key}

    params = {
        "lat": lat,
        "lon": lon,
        "hours": hours_ahead,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{settings.metservice_base_url}/pointforecast",
            params=params,
            headers=headers,
        )
        response.raise_for_status()

        data = response.json()
        forecasts = []

        for item in data.get("forecasts", []):
            forecasts.append(
                WeatherForecast(
                    timestamp=datetime.fromisoformat(item["time"]),
                    wind_speed_ms=item.get("wind_speed", 0),
                    wind_direction_deg=item.get("wind_direction", 0),
                    wind_gust_ms=item.get("wind_gust"),
                    temperature_c=item.get("temperature"),
                    pressure_hpa=item.get("pressure"),
                    humidity_pct=item.get("humidity"),
                )
            )

        return forecasts


def _generate_synthetic_weather(
    hours_ahead: int,
    interval_hours: int = 1,
) -> list[WeatherForecast]:
    """Generate synthetic weather data for development.

    Creates realistic Bay of Islands wind patterns:
    - Morning calm
    - Afternoon sea breeze (NE-E)
    - Evening wind drop
    - Occasional frontal systems
    """
    forecasts = []
    current_time = datetime.now()
    rng = np.random.default_rng(789)

    # Whether we're in a "windy event"
    in_event = False
    event_hours_remaining = 0
    event_base_speed = 0.0
    event_direction = 0.0

    for i in range(0, hours_ahead, interval_hours):
        timestamp = current_time + timedelta(hours=i)
        hour = timestamp.hour

        if in_event:
            event_hours_remaining -= interval_hours
            if event_hours_remaining <= 0:
                in_event = False
            else:
                # Event wind
                base_speed = event_base_speed * (0.8 + 0.4 * rng.random())
                direction = (event_direction + rng.uniform(-20, 20)) % 360
        else:
            # Normal diurnal pattern
            # Morning (5-10): Light and variable
            if 5 <= hour < 10:
                base_speed = 2.0 + rng.uniform(-1, 2)
                direction = rng.uniform(0, 360)
            # Midday-afternoon (10-18): Sea breeze from NE-E
            elif 10 <= hour < 18:
                # Peak around 14:00
                sea_breeze_factor = 1 - abs(hour - 14) / 4
                base_speed = 4.0 + 4.0 * sea_breeze_factor + rng.uniform(-1, 2)
                direction = 45 + rng.uniform(-30, 30)  # NE with variation
            # Evening (18-22): Dying sea breeze
            elif 18 <= hour < 22:
                base_speed = 3.0 - (hour - 18) * 0.5 + rng.uniform(-1, 1)
                direction = 60 + rng.uniform(-40, 40)
            # Night (22-5): Light and variable
            else:
                base_speed = 1.5 + rng.uniform(-0.5, 1)
                direction = rng.uniform(0, 360)

            # Random chance of starting a windy event (front)
            if rng.random() < 0.02:
                in_event = True
                event_hours_remaining = int(rng.uniform(6, 24))
                event_base_speed = rng.uniform(8, 15)
                event_direction = rng.uniform(180, 270)  # SW-W fronts
                base_speed = event_base_speed
                direction = event_direction

        # Add gusts
        gust = base_speed * (1.3 + 0.3 * rng.random()) if base_speed > 3 else None

        forecasts.append(
            WeatherForecast(
                timestamp=timestamp,
                wind_speed_ms=round(max(0, base_speed), 1),
                wind_direction_deg=round(direction % 360, 0),
                wind_gust_ms=round(gust, 1) if gust else None,
                temperature_c=round(15 + 5 * np.sin(np.pi * hour / 12) + rng.uniform(-2, 2), 1),
                confidence=0.9 - 0.3 * (i / hours_ahead),
            )
        )

    return forecasts


def wind_effect_on_kayak(
    wind_speed_ms: float,
    wind_direction_deg: float,
    kayak_heading_deg: float,
    drag_coefficient: float = 0.02,
    frontal_area: float = 0.4,
) -> tuple[float, float]:
    """Calculate wind effect on kayak as velocity components.

    Args:
        wind_speed_ms: Wind speed in m/s.
        wind_direction_deg: Wind direction (from) in degrees.
        kayak_heading_deg: Kayak heading in degrees.
        drag_coefficient: Kayak wind drag coefficient.
        frontal_area: Effective frontal area in m².

    Returns:
        Tuple of (eastward_velocity, northward_velocity) in m/s.
    """
    # Wind direction is "from", so wind vector points opposite
    wind_to_rad = np.radians(90 - wind_direction_deg + 180)

    # Wind force ~ C_d * A * rho * v^2 / 2
    # Resulting velocity ~ C_d * A * rho * v^2 / (2 * mass)
    # Simplified: v_effect = k * v_wind^2 where k absorbs constants
    # For a kayak: typical k ~ 0.01-0.03

    force_factor = drag_coefficient * frontal_area * wind_speed_ms
    # Velocity effect (simplified model assuming low-speed equilibrium)
    v_effect = force_factor * wind_speed_ms * 0.1

    vx = v_effect * np.cos(wind_to_rad)
    vy = v_effect * np.sin(wind_to_rad)

    return float(vx), float(vy)
