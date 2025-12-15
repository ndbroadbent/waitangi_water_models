"""Data classes for pipeline communication."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters."""

    # Grid parameters
    downsample: int = 8
    dx: float = 8.0
    dy: float = 8.0

    # Physical parameters
    g: float = 9.81
    max_vel: float = 3.0

    # Tidal parameters
    low_tide: float = -0.5
    high_tide: float = 1.1
    tide_period: float = 12.42 * 3600  # M2 tide in seconds

    # River parameters
    river_flow: float = 1.0  # m³/s
    tracer_diffusion: float = 0.5

    # Simulation timing
    duration_hours: float | None = None  # None = one full tidal cycle
    output_interval: float = 300.0  # seconds between frames
    skip_equilibrium: bool = False  # Skip initial equilibration phase
    fixed_tide: float | None = None  # Fixed tide level (disables tidal cycle)

    # Wall/boundary locations
    wall_col: int = 382
    wall_north_row: int = 124

    # River source location
    falls_row: int = 214
    falls_col: int = 24
    river_radius: int = 3

    # Manning's n coefficients
    manning_open_water: float = 0.035
    manning_mangrove: float = 0.12

    @property
    def mean_tide(self) -> float:
        return (self.low_tide + self.high_tide) / 2

    @property
    def tide_amplitude(self) -> float:
        return (self.high_tide - self.low_tide) / 2


@dataclass
class RenderConfig:
    """Configuration for frame rendering."""

    # Figure size
    fig_width: float = 14
    fig_height: float = 10
    dpi: int = 100

    # Quiver plot settings
    quiver_skip: int = 10
    quiver_scale: float = 30.0
    quiver_alpha: float = 0.4

    # Kayak display
    kayak_scale: float = 25  # meters for display

    # Basemap
    use_basemap: bool = True
    basemap_zoom: int = 15

    # Video settings
    framerate: int = 10
    crf: int = 18  # Quality (lower = better)


@dataclass
class GaugeData:
    """Flow gauge measurement at a single point."""

    name: str
    depth: float
    speed: float
    flow: float
    direction: str
    x: float
    y: float


@dataclass
class KayakState:
    """Current kayak position and orientation."""

    x: float
    y: float
    heading: float  # degrees, 0=East, 90=North


@dataclass
class FrameData:
    """All data needed to render a single frame.

    This is passed from the simulation thread to render workers.
    Contains numpy arrays (not JAX arrays) for thread safety.
    """

    # Frame identification
    frame_number: int
    total_frames: int

    # Timing
    simulation_time: float  # seconds
    tide_level: float

    # Grid state (numpy arrays, already transferred from GPU)
    h: "NDArray[np.float32]"  # Water depth
    u: "NDArray[np.float32]"  # Velocity x-component
    v: "NDArray[np.float32]"  # Velocity y-component
    river_tracer: "NDArray[np.float32]"  # River tracer concentration
    ocean_tracer: "NDArray[np.float32]"  # Ocean tracer concentration
    z_bed: "NDArray[np.float32]"  # Bed elevation (static)

    # Computed metrics
    wet_area_km2: float

    # Gauge measurements
    gauges: list[GaugeData]

    # Kayak state
    kayak: KayakState

    # Configuration reference
    river_flow: float


@dataclass
class RenderedFrame:
    """A rendered frame ready for video encoding.

    Contains PNG image data and frame number for ordering.
    """

    frame_number: int
    png_data: bytes


# Sentinel value to signal end of queue
END_OF_STREAM = object()
