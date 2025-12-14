"""Water tracer simulation for visualizing river/tidal mixing.

Tracks concentration of "river water" (green dye) as it flows from
Haruru Falls through the estuary and out to sea.
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from pyproj import Transformer

from waitangi.data.elevation import ElevationData, compute_flooded_area_at_level


@dataclass
class TracerField:
    """2D tracer concentration field.

    Values represent fraction of river water (0 = pure sea water, 1 = pure river water).
    """
    concentration: np.ndarray  # Shape: (height, width), values 0-1
    water_mask: np.ndarray     # Boolean mask of flooded cells
    transform: object          # Affine transform for georeferencing
    timestamp: datetime
    water_level: float         # Current water level (m)


class WaterTracer:
    """Simulates river water tracer through the estuary.

    Uses a simple advection-diffusion model on a 2D grid derived from
    the elevation data.
    """

    def __init__(self, elevation_data: ElevationData, resolution_factor: int = 4):
        """Initialize tracer simulation.

        Args:
            elevation_data: LiDAR elevation data
            resolution_factor: Downsampling factor for simulation grid (4 = 1/4 resolution)
        """
        self.elev = elevation_data
        self.resolution_factor = resolution_factor

        # Downsample elevation for faster simulation
        self.elev_grid = elevation_data.data[::resolution_factor, ::resolution_factor]
        self.grid_shape = self.elev_grid.shape

        # Pixel size in meters (after downsampling)
        self.pixel_size = abs(elevation_data.transform.a) * resolution_factor

        # Create adjusted transform for downsampled grid
        self.transform = elevation_data.transform

        # River source location (Haruru Falls area)
        self._init_source_location()

        # Ocean boundary (eastern edge of domain)
        self._init_ocean_boundary()

    def _init_source_location(self):
        """Set up river inflow source at Haruru Falls."""
        # Haruru Falls coordinates
        haruru_lat, haruru_lon = -35.278284, 174.051297

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
        easting, northing = transformer.transform(haruru_lon, haruru_lat)

        # Convert to grid coordinates
        col = int((easting - self.elev.transform.c) / self.elev.transform.a) // self.resolution_factor
        row = int((northing - self.elev.transform.f) / self.elev.transform.e) // self.resolution_factor

        # Source region (small area around falls)
        self.source_row = max(0, min(row, self.grid_shape[0] - 1))
        self.source_col = max(0, min(col, self.grid_shape[1] - 1))
        self.source_radius = 3  # cells

    def _init_ocean_boundary(self):
        """Set up ocean boundary on eastern side."""
        # Ocean is on the right side of the domain
        self.ocean_cols = slice(-10, None)  # Last 10 columns

    def compute_velocity_field(self, water_level: float, river_flow_m3s: float,
                                tide_rate_m_per_hour: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute water velocity field based on bathymetry and conditions.

        Args:
            water_level: Current water level (m above datum)
            river_flow_m3s: River discharge (m³/s)
            tide_rate_m_per_hour: Rate of tide change (positive = rising)

        Returns:
            (u, v) velocity arrays in m/s (eastward, northward)
        """
        # Get flooded area at current water level
        _, water_mask_full = compute_flooded_area_at_level(water_level, self.elev)
        water_mask = water_mask_full[::self.resolution_factor, ::self.resolution_factor]

        # Water depth
        depth = np.maximum(water_level - self.elev_grid, 0.01)  # Min 1cm to avoid div/0
        depth = np.where(water_mask, depth, 0.01)

        # Initialize velocity fields
        u = np.zeros(self.grid_shape)  # Eastward velocity
        v = np.zeros(self.grid_shape)  # Northward velocity

        # 1. Compute gradient-driven flow (water flows downhill/toward deeper water)
        # Use elevation gradient to determine flow direction
        grad_y, grad_x = np.gradient(self.elev_grid)

        # Normalize gradient
        grad_mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-6
        grad_x_norm = -grad_x / grad_mag  # Negative because flow is downhill
        grad_y_norm = -grad_y / grad_mag

        # 2. River inflow component
        # River water enters at source and flows generally eastward (downstream)
        # Velocity based on continuity: Q = v * A, v = Q / (depth * width)
        channel_width = 50  # Approximate channel width in meters
        river_velocity = river_flow_m3s / (depth * channel_width + 0.1)
        river_velocity = np.clip(river_velocity, 0, 2.0)  # Max 2 m/s

        # River flows generally east-northeast toward the bridge
        river_dir_x = 0.9  # Mostly eastward
        river_dir_y = 0.3  # Slightly northward

        # Weight river influence by proximity to source and depth
        dist_from_source = np.sqrt(
            (np.arange(self.grid_shape[1]) - self.source_col)**2 +
            (np.arange(self.grid_shape[0])[:, np.newaxis] - self.source_row)**2
        )
        river_weight = np.exp(-dist_from_source / 100)  # Decay over ~100 cells

        u += river_velocity * river_dir_x * river_weight
        v += river_velocity * river_dir_y * river_weight

        # 3. Tidal component
        # Rising tide: water flows INTO estuary (westward from ocean)
        # Falling tide: water flows OUT of estuary (eastward to ocean)
        tide_velocity = tide_rate_m_per_hour / 3600 * 1000  # Convert to m/s equivalent flow
        tide_velocity = np.clip(tide_velocity, -0.5, 0.5)

        # Tidal flow is stronger in deeper channels
        channel_factor = np.clip(depth / 2.0, 0, 1)

        # Distance from ocean affects tidal influence
        dist_from_ocean = self.grid_shape[1] - np.arange(self.grid_shape[1])
        ocean_influence = np.exp(-dist_from_ocean / 200)[np.newaxis, :]

        # Negative tide_velocity = falling tide = flow toward ocean (positive x)
        u += -tide_velocity * channel_factor * ocean_influence * 10

        # 4. Apply water mask - no velocity on land
        u = np.where(water_mask, u, 0)
        v = np.where(water_mask, v, 0)

        # 5. Add some bathymetric steering (water follows channels)
        u += grad_x_norm * 0.1 * channel_factor
        v += grad_y_norm * 0.1 * channel_factor

        return u, v, water_mask

    def create_initial_state(self, water_level: float) -> TracerField:
        """Create initial tracer field with river source.

        Args:
            water_level: Initial water level (m)

        Returns:
            Initial tracer field
        """
        _, water_mask_full = compute_flooded_area_at_level(water_level, self.elev)
        water_mask = water_mask_full[::self.resolution_factor, ::self.resolution_factor]

        # Start with all seawater (concentration = 0)
        concentration = np.zeros(self.grid_shape)

        # Add river water at source
        row_slice = slice(
            max(0, self.source_row - self.source_radius),
            min(self.grid_shape[0], self.source_row + self.source_radius + 1)
        )
        col_slice = slice(
            max(0, self.source_col - self.source_radius),
            min(self.grid_shape[1], self.source_col + self.source_radius + 1)
        )
        concentration[row_slice, col_slice] = 1.0

        return TracerField(
            concentration=concentration,
            water_mask=water_mask,
            transform=self.transform,
            timestamp=datetime.now(),
            water_level=water_level,
        )

    def step(self, state: TracerField, dt_seconds: float,
             river_flow_m3s: float, tide_rate_m_per_hour: float,
             diffusion_coeff: float = 5.0) -> TracerField:
        """Advance tracer field by one time step.

        Args:
            state: Current tracer state
            dt_seconds: Time step in seconds
            river_flow_m3s: River discharge
            tide_rate_m_per_hour: Tide change rate
            diffusion_coeff: Diffusion coefficient (m²/s)

        Returns:
            Updated tracer field
        """
        # Get velocity field
        u, v, water_mask = self.compute_velocity_field(
            state.water_level, river_flow_m3s, tide_rate_m_per_hour
        )

        conc = state.concentration.copy()

        # Advection using upwind scheme
        # Eastward flux
        u_pos = np.maximum(u, 0)
        u_neg = np.minimum(u, 0)

        # Northward flux (note: row index increases southward in image coords)
        v_pos = np.maximum(v, 0)
        v_neg = np.minimum(v, 0)

        # Upwind differences
        dconc_dx_pos = np.diff(conc, axis=1, prepend=conc[:, :1])
        dconc_dx_neg = np.diff(conc, axis=1, append=conc[:, -1:])
        dconc_dy_pos = np.diff(conc, axis=0, prepend=conc[:1, :])
        dconc_dy_neg = np.diff(conc, axis=0, append=conc[-1:, :])

        # Advection term
        advection = (
            u_pos * dconc_dx_pos / self.pixel_size +
            u_neg * dconc_dx_neg / self.pixel_size +
            v_pos * dconc_dy_pos / self.pixel_size +
            v_neg * dconc_dy_neg / self.pixel_size
        )

        # Diffusion (Laplacian)
        laplacian = (
            np.roll(conc, 1, axis=0) + np.roll(conc, -1, axis=0) +
            np.roll(conc, 1, axis=1) + np.roll(conc, -1, axis=1) -
            4 * conc
        ) / (self.pixel_size ** 2)

        # Update concentration
        conc_new = conc - advection * dt_seconds + diffusion_coeff * laplacian * dt_seconds

        # Enforce boundary conditions
        # River source: constant concentration = 1
        row_slice = slice(
            max(0, self.source_row - self.source_radius),
            min(self.grid_shape[0], self.source_row + self.source_radius + 1)
        )
        col_slice = slice(
            max(0, self.source_col - self.source_radius),
            min(self.grid_shape[1], self.source_col + self.source_radius + 1)
        )
        conc_new[row_slice, col_slice] = 1.0

        # Ocean boundary: concentration = 0 (pure seawater)
        conc_new[:, self.ocean_cols] = 0.0

        # Clip to valid range and apply water mask
        conc_new = np.clip(conc_new, 0, 1)
        conc_new = np.where(water_mask, conc_new, 0)

        return TracerField(
            concentration=conc_new,
            water_mask=water_mask,
            transform=self.transform,
            timestamp=state.timestamp + timedelta(seconds=dt_seconds),
            water_level=state.water_level,
        )

    def run_simulation(
        self,
        duration_hours: float,
        dt_seconds: float = 60.0,
        river_flow_m3s: float = 0.5,
        tide_range_m: float = 1.6,
        tide_period_hours: float = 12.42,
        start_tide_phase: float = 0.0,  # 0 = low tide, 0.5 = high tide
        output_interval_minutes: float = 5.0,
    ) -> list[TracerField]:
        """Run full tracer simulation.

        Args:
            duration_hours: Simulation duration
            dt_seconds: Time step
            river_flow_m3s: Constant river discharge
            tide_range_m: Tidal range (high - low)
            tide_period_hours: Tidal period (default M2 = 12.42 hours)
            start_tide_phase: Starting phase (0=low, 0.5=high, 1=low)
            output_interval_minutes: How often to save frames

        Returns:
            List of tracer fields at output intervals
        """
        # Tidal parameters
        mean_water_level = 0.3  # Mean sea level offset
        tide_amplitude = tide_range_m / 2
        omega = 2 * np.pi / (tide_period_hours * 3600)  # rad/s

        # Initial water level
        t = 0
        phase = start_tide_phase * 2 * np.pi
        water_level = mean_water_level + tide_amplitude * np.sin(phase)

        # Initialize
        state = self.create_initial_state(water_level)
        results = [state]

        # Simulation loop
        n_steps = int(duration_hours * 3600 / dt_seconds)
        output_every = int(output_interval_minutes * 60 / dt_seconds)

        for i in range(n_steps):
            t = i * dt_seconds

            # Update water level (sinusoidal tide)
            water_level = mean_water_level + tide_amplitude * np.sin(omega * t + phase)
            state = TracerField(
                concentration=state.concentration,
                water_mask=state.water_mask,
                transform=state.transform,
                timestamp=state.timestamp,
                water_level=water_level,
            )

            # Tide rate (derivative of water level)
            tide_rate = tide_amplitude * omega * np.cos(omega * t + phase) * 3600  # m/hour

            # Step simulation
            state = self.step(state, dt_seconds, river_flow_m3s, tide_rate)

            # Save output
            if (i + 1) % output_every == 0:
                results.append(state)

        return results
