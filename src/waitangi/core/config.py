"""Configuration and settings for the simulation system."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PlanningHorizon(str, Enum):
    """Planning horizon determines which data sources drive the simulation."""

    NOWCAST = "nowcast"  # 0-12h: gauge-driven
    NEAR_TERM = "near_term"  # 12h-7d: rainfall forecast + gauge
    MEDIUM = "medium"  # 1-4 weeks: scenario-based


class GPUBackend(str, Enum):
    """GPU compute backend selection."""

    CPU = "cpu"  # JAX on CPU (fallback)
    CUDA = "cuda"  # NVIDIA GPU
    METAL = "metal"  # Apple Silicon


class WaitangiLocation:
    """Geographic constants for the Waitangi River system."""

    # Mouth of Waitangi River (Bay of Islands)
    MOUTH_LAT = -35.2675
    MOUTH_LON = 174.0858

    # Approximate upstream extent for simulation
    UPSTREAM_LAT = -35.2900
    UPSTREAM_LON = 174.0650

    # NZTM2000 projection EPSG code
    CRS_NZTM = "EPSG:2193"
    CRS_WGS84 = "EPSG:4326"

    # Tide gauge reference port (Opua)
    TIDE_PORT = "Opua"
    TIDE_PORT_LAT = -35.3122
    TIDE_PORT_LON = 174.1219


class TideSettings(BaseSettings):
    """Tide model configuration."""

    model_config = SettingsConfigDict(env_prefix="TIDE_")

    # Harmonic constituents to model
    use_m2: bool = True  # Principal lunar semidiurnal
    use_s2: bool = True  # Principal solar semidiurnal
    use_n2: bool = True  # Larger lunar elliptic
    use_k1: bool = True  # Lunar diurnal
    use_o1: bool = True  # Lunar diurnal

    # Mean sea level (m above chart datum)
    mean_sea_level: float = 1.5

    # Spring/neap amplitude range (m)
    spring_amplitude: float = 1.4
    neap_amplitude: float = 0.8

    # Phase lag from reference port to river mouth (minutes)
    phase_lag_minutes: float = 15.0


class RiverSettings(BaseSettings):
    """River discharge model configuration."""

    model_config = SettingsConfigDict(env_prefix="RIVER_")

    # Baseline dry-weather flow (m³/s)
    q_base: float = 2.0

    # Runoff coefficient (dimensionless, 0-1)
    runoff_coefficient: float = 0.3

    # Response time constant (hours)
    tau_hours: float = 6.0

    # Lag time from rainfall to flow (hours)
    delay_hours: float = 2.0

    # Catchment area (km²)
    catchment_area_km2: float = 85.0

    # Decay length for velocity (m) - how far upstream tide penetrates
    decay_length_m: float = 3000.0

    # Cross-sectional area at gauge (m²) for Q -> V conversion
    cross_section_m2: float = 25.0


class WindSettings(BaseSettings):
    """Wind effect model configuration."""

    model_config = SettingsConfigDict(env_prefix="WIND_")

    # Wind drag coefficient on kayak (dimensionless)
    drag_coefficient: float = 0.02

    # Reference height for wind speed (m)
    reference_height: float = 10.0

    # Kayak frontal area (m²)
    kayak_frontal_area: float = 0.4


class KayakSettings(BaseSettings):
    """Kayak and paddling configuration."""

    model_config = SettingsConfigDict(env_prefix="KAYAK_")

    # Maximum sustainable paddling speed (m/s) - ~6 km/h
    max_paddle_speed: float = 1.67

    # Typical cruising speed (m/s) - ~4 km/h
    cruise_paddle_speed: float = 1.11

    # Sprint speed (m/s) - ~8 km/h
    sprint_paddle_speed: float = 2.22

    # Drag coefficient in water (dimensionless)
    water_drag: float = 0.01


class SimulationSettings(BaseSettings):
    """Simulation loop configuration."""

    model_config = SettingsConfigDict(env_prefix="SIM_")

    # Time step (seconds)
    dt: float = 1.0

    # Maximum simulation duration (hours)
    max_duration_hours: float = 12.0

    # Output interval (seconds)
    output_interval: float = 60.0

    # Integration method
    use_rk2: bool = True  # False = Euler

    # GPU batch size for particles
    particle_batch_size: int = 10000


class DataSourceSettings(BaseSettings):
    """External data source configuration."""

    model_config = SettingsConfigDict()

    # NRC Environmental Data Hub
    nrc_base_url: str = "https://envdata.nrc.govt.nz"

    # MetService API (requires subscription)
    metservice_api_key: str = ""
    metservice_base_url: str = "https://api.metservice.com/v1"

    # LINZ Data Service
    linz_api_key: str = ""
    linz_base_url: str = "https://data.linz.govt.nz"

    # NIWA tide predictions
    niwa_base_url: str = "https://tides.niwa.co.nz"

    # Cache directory
    cache_dir: Path = Path("~/.cache/waitangi").expanduser()

    # Update intervals (minutes)
    gauge_update_interval: int = 15
    weather_update_interval: int = 60


class Settings(BaseSettings):
    """Master configuration aggregating all subsystems."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Subsystem settings
    tide: TideSettings = Field(default_factory=TideSettings)
    river: RiverSettings = Field(default_factory=RiverSettings)
    wind: WindSettings = Field(default_factory=WindSettings)
    kayak: KayakSettings = Field(default_factory=KayakSettings)
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    data_sources: DataSourceSettings = Field(default_factory=DataSourceSettings)

    # GPU backend
    gpu_backend: GPUBackend = GPUBackend.CPU

    # Planning horizon
    planning_horizon: PlanningHorizon = PlanningHorizon.NOWCAST

    # Simulation start time (defaults to now)
    start_time: datetime | None = None

    # Debug mode
    debug: bool = False

    @model_validator(mode="after")
    def set_defaults(self) -> Self:
        """Set default start time if not provided."""
        if self.start_time is None:
            self.start_time = datetime.now()
        return self


def get_settings() -> Settings:
    """Load settings from environment and .env file."""
    return Settings()
