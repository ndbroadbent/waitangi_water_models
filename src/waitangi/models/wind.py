"""Wind effect model for kayak motion.

Wind affects the kayak directly (not the water) through aerodynamic drag.
"""

from dataclasses import dataclass
from datetime import datetime

import jax.numpy as jnp
import numpy as np
from jax import Array

from waitangi.core.config import WindSettings, get_settings
from waitangi.core.constants import RHO_AIR
from waitangi.data.weather import WeatherData


@dataclass
class WindModel:
    """Model for wind effects on kayak motion.

    The wind imparts a force on the kayak:
    F_wind = 0.5 * rho * C_d * A * v_wind^2

    This is converted to an effective velocity based on drag equilibrium.
    """

    settings: WindSettings

    # Cached weather data
    _weather_data: WeatherData | None = None

    @classmethod
    def create(cls, settings: WindSettings | None = None) -> "WindModel":
        """Create a wind model with default settings."""
        if settings is None:
            settings = get_settings().wind
        return cls(settings=settings)

    def ingest_weather_data(self, weather_data: WeatherData) -> None:
        """Ingest weather forecast data.

        Args:
            weather_data: Weather forecast time series.
        """
        self._weather_data = weather_data

    def get_wind(self, timestamp: datetime) -> tuple[float, float]:
        """Get wind speed and direction at given time.

        Args:
            timestamp: Time for wind lookup.

        Returns:
            Tuple of (speed_ms, direction_deg).
        """
        if self._weather_data:
            return self._weather_data.interpolate_wind(timestamp)
        return 0.0, 0.0

    def get_wind_components(self, timestamp: datetime) -> tuple[float, float]:
        """Get wind velocity components.

        Args:
            timestamp: Time for wind lookup.

        Returns:
            Tuple of (eastward_ms, northward_ms).
        """
        speed, direction = self.get_wind(timestamp)

        # Convert from meteorological (from direction) to velocity components
        # Wind direction is "from", so vector points opposite
        direction_rad = np.radians(90 - direction + 180)

        u = speed * np.cos(direction_rad)
        v = speed * np.sin(direction_rad)

        return float(u), float(v)

    def get_kayak_drift(
        self,
        timestamp: datetime,
        kayak_heading_deg: float | None = None,
    ) -> tuple[float, float]:
        """Calculate wind-induced drift velocity for kayak.

        The drift velocity depends on wind speed squared (drag force)
        and the kayak's orientation relative to the wind.

        Args:
            timestamp: Time for calculation.
            kayak_heading_deg: Kayak heading in degrees (from north).
                              If None, uses full broadside exposure.

        Returns:
            Tuple of (eastward_drift_ms, northward_drift_ms).
        """
        speed, direction = self.get_wind(timestamp)

        if speed < 0.1:
            return 0.0, 0.0

        # Calculate drag force
        drag_force = (
            0.5
            * RHO_AIR
            * self.settings.drag_coefficient
            * self.settings.kayak_frontal_area
            * speed**2
        )

        # Convert force to drift velocity
        # Using simple equilibrium: F_wind = F_drag_water
        # F_drag_water ~ C_d_water * v_drift^2
        # Assume water drag coefficient is much larger than air
        # Simplified: v_drift ~ k * v_wind^2 where k is empirical
        k = 0.005  # Empirical drift coefficient

        drift_speed = k * speed**2

        # Heading factor (broadside = 1, head-on = 0.3)
        if kayak_heading_deg is not None:
            relative_angle = abs((direction - kayak_heading_deg + 180) % 360 - 180)
            # Maximum drift when wind is perpendicular (90°)
            heading_factor = 0.3 + 0.7 * abs(np.sin(np.radians(relative_angle)))
            drift_speed *= heading_factor

        # Wind direction is "from", drift is "to"
        direction_rad = np.radians(90 - direction + 180)

        drift_u = drift_speed * np.cos(direction_rad)
        drift_v = drift_speed * np.sin(direction_rad)

        return float(drift_u), float(drift_v)

    def get_kayak_drift_field(
        self,
        timestamp: datetime,
        x: Array,
        y: Array,
    ) -> tuple[Array, Array]:
        """Get wind drift velocity field.

        For now, assumes uniform wind across the domain.
        Could be extended with spatial variation.

        Args:
            timestamp: Time for calculation.
            x: Array of x coordinates.
            y: Array of y coordinates.

        Returns:
            Tuple of (u_drift, v_drift) arrays in m/s.
        """
        drift_u, drift_v = self.get_kayak_drift(timestamp)

        # Uniform field
        u = jnp.full_like(x, drift_u)
        v = jnp.full_like(y, drift_v)

        return u, v

    def get_wind_description(self, timestamp: datetime) -> str:
        """Get human-readable wind description.

        Args:
            timestamp: Time for description.

        Returns:
            String like "Light NE breeze (5 m/s)".
        """
        speed, direction = self.get_wind(timestamp)

        # Beaufort scale descriptions
        if speed < 0.5:
            strength = "Calm"
        elif speed < 1.5:
            strength = "Light air"
        elif speed < 3.5:
            strength = "Light breeze"
        elif speed < 5.5:
            strength = "Gentle breeze"
        elif speed < 8.0:
            strength = "Moderate breeze"
        elif speed < 10.5:
            strength = "Fresh breeze"
        elif speed < 14.0:
            strength = "Strong breeze"
        else:
            strength = "High wind"

        # Cardinal direction
        directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                     "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        idx = int((direction + 11.25) / 22.5) % 16
        cardinal = directions[idx]

        return f"{strength} from {cardinal} ({speed:.1f} m/s)"
