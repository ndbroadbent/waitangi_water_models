"""GPU simulation engine using JAX Metal backend.

Runs the shallow water simulation on GPU and produces frame data
for the rendering pipeline.
"""

import json
import threading
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from pyproj import Transformer
from shapely.geometry import Point, Polygon, shape
from shapely.ops import unary_union

from waitangi.data.elevation import fetch_waitangi_elevation
from waitangi.pipeline.data import (
    END_OF_STREAM,
    FrameData,
    GaugeData,
    KayakState,
    SimulationConfig,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Use 32-bit precision (required for Metal GPU backend)
jax.config.update("jax_enable_x64", False)

# Kayak parameters
BOAT_RAMP_LAT = -35.270801
BOAT_RAMP_LON = 174.078956


def _velocity_to_compass(u: float, v: float) -> str:
    """Convert u,v velocity components to compass direction."""
    speed = np.sqrt(u**2 + v**2)
    if speed < 0.001:
        return "--"
    angle = np.degrees(np.arctan2(v, u))
    bearing = (90 - angle) % 360
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((bearing + 22.5) / 45) % 8
    return directions[idx]


def _interpolate_velocity(
    x: float,
    y: float,
    u: "NDArray",
    v: "NDArray",
    extent: tuple,
    ny: int,
    nx: int,
) -> tuple[float, float]:
    """Bilinear interpolation of velocity at a point."""
    xmin, xmax, ymin, ymax = extent
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    col = (x - xmin) / dx
    row = (ymax - y) / dy

    if row < 0 or row >= ny - 1 or col < 0 or col >= nx - 1:
        return 0.0, 0.0

    r0, c0 = int(row), int(col)
    r1, c1 = r0 + 1, c0 + 1
    fr, fc = row - r0, col - c0

    u_interp = (
        u[r0, c0] * (1 - fr) * (1 - fc)
        + u[r0, c1] * (1 - fr) * fc
        + u[r1, c0] * fr * (1 - fc)
        + u[r1, c1] * fr * fc
    )
    v_interp = (
        v[r0, c0] * (1 - fr) * (1 - fc)
        + v[r0, c1] * (1 - fr) * fc
        + v[r1, c0] * fr * (1 - fc)
        + v[r1, c1] * fr * fc
    )

    return float(u_interp), float(v_interp)


@jit
def _advect_tracer(
    tracer, h, h_new, u, v, h_e, h_n, dx, dy, dt, tracer_diffusion, source_mask, source_value
):
    """Advect and diffuse a single tracer field."""
    tracer = jnp.where(source_mask, source_value, tracer)

    wet_old = h > 0.01
    wet_new = h_new > 0.01

    tracer_e = jnp.where(u[:-1, :] > 0, tracer[:-1, :], tracer[1:, :])
    tracer_n = jnp.where(v[:, :-1] > 0, tracer[:, :-1], tracer[:, 1:])

    tracer_flux_e = u[:-1, :] * h_e * tracer_e
    tracer_flux_n = v[:, :-1] * h_n * tracer_n

    tracer_div = jnp.zeros_like(tracer)
    tracer_div = tracer_div.at[:-1, :].add(tracer_flux_e / dx)
    tracer_div = tracer_div.at[1:, :].add(-tracer_flux_e / dx)
    tracer_div = tracer_div.at[:, :-1].add(tracer_flux_n / dy)
    tracer_div = tracer_div.at[:, 1:].add(-tracer_flux_n / dy)

    h_safe = jnp.maximum(h_new, 0.01)

    # Standard advection for cells that were already wet
    tracer_advected = jnp.where(wet_new, (tracer * h - dt * tracer_div) / h_safe, 0.0)
    tracer_advected = jnp.clip(tracer_advected, 0.0, 1.0)

    # For newly wet cells (dry -> wet), inherit tracer from wet neighbors
    # This ensures water spreading onto dry land carries its tracer
    newly_wet = wet_new & ~wet_old
    tracer_padded = jnp.pad(tracer, 1, mode="constant", constant_values=0.0)
    wet_padded = jnp.pad(wet_old, 1, mode="constant", constant_values=False)

    # Get tracer values from 4 neighbors
    neighbor_tracer_sum = (
        tracer_padded[:-2, 1:-1] + tracer_padded[2:, 1:-1] +
        tracer_padded[1:-1, :-2] + tracer_padded[1:-1, 2:]
    )
    neighbor_wet_count = (
        wet_padded[:-2, 1:-1].astype(jnp.float32) + wet_padded[2:, 1:-1].astype(jnp.float32) +
        wet_padded[1:-1, :-2].astype(jnp.float32) + wet_padded[1:-1, 2:].astype(jnp.float32)
    )
    neighbor_tracer_avg = jnp.where(neighbor_wet_count > 0, neighbor_tracer_sum / neighbor_wet_count, 0.0)

    # Use neighbor average for newly wet cells, advected value otherwise
    tracer = jnp.where(newly_wet, neighbor_tracer_avg, tracer_advected)

    # Diffusion
    tracer_padded2 = jnp.pad(tracer, 1, mode="constant", constant_values=0.0)
    lap = (
        tracer_padded2[:-2, 1:-1]
        + tracer_padded2[2:, 1:-1]
        + tracer_padded2[1:-1, :-2]
        + tracer_padded2[1:-1, 2:]
        - 4 * tracer
    ) / (dx * dy)
    tracer = tracer + tracer_diffusion * dt * lap
    tracer = jnp.clip(tracer, 0.0, 1.0)

    tracer = jnp.where(wet_new, tracer, 0.0)
    tracer = jnp.where(source_mask & wet_new, source_value, tracer)

    return tracer


@jit
def _simulation_step(
    eta,
    u,
    v,
    river_tracer,
    ocean_tracer,
    z_bed,
    manning_field,
    params,
    river_mask,
    ocean_inlet_mask,
    tracer_diffusion: float = 5.0,
):
    """Single simulation timestep - JIT compiled for GPU."""
    dx, dy, dt, g, max_vel = params

    h = jnp.maximum(eta - z_bed, 0.0)

    deta_dx = jnp.zeros_like(eta)
    deta_dx = deta_dx.at[:-1, :].set((eta[1:, :] - eta[:-1, :]) / dx)

    deta_dy = jnp.zeros_like(eta)
    deta_dy = deta_dy.at[:, :-1].set((eta[:, 1:] - eta[:, :-1]) / dy)

    u = u.at[:-1, :].add(-g * dt * deta_dx[:-1, :])
    v = v.at[:, :-1].add(-g * dt * deta_dy[:, :-1])

    h_at_u = jnp.maximum((h[:-1, :] + h[1:, :]) / 2, 0.01)
    h_at_v = jnp.maximum((h[:, :-1] + h[:, 1:]) / 2, 0.01)

    manning_at_u = (manning_field[:-1, :] + manning_field[1:, :]) / 2
    manning_at_v = (manning_field[:, :-1] + manning_field[:, 1:]) / 2

    Cf_u = g * manning_at_u**2 / jnp.power(h_at_u, 1 / 3)
    Cf_v = g * manning_at_v**2 / jnp.power(h_at_v, 1 / 3)

    friction_u = 1.0 / (1.0 + dt * Cf_u * jnp.abs(u[:-1, :]) / h_at_u)
    friction_v = 1.0 / (1.0 + dt * Cf_v * jnp.abs(v[:, :-1]) / h_at_v)

    u = u.at[:-1, :].multiply(friction_u)
    v = v.at[:, :-1].multiply(friction_v)

    u = jnp.clip(u, -max_vel, max_vel)
    v = jnp.clip(v, -max_vel, max_vel)

    u = u.at[-1, :].set(0.0)
    u = u.at[0, :].set(0.0)
    v = v.at[:, -1].set(0.0)
    v = v.at[:, 0].set(0.0)

    h_e = jnp.where(u[:-1, :] > 0, h[:-1, :], h[1:, :])
    h_n = jnp.where(v[:, :-1] > 0, h[:, :-1], h[:, 1:])

    flux_e = u[:-1, :] * h_e
    flux_n = v[:, :-1] * h_n

    div = jnp.zeros_like(eta)
    div = div.at[:-1, :].add(flux_e / dx)
    div = div.at[1:, :].add(-flux_e / dx)
    div = div.at[:, :-1].add(flux_n / dy)
    div = div.at[:, 1:].add(-flux_n / dy)

    eta = eta - dt * div
    eta = jnp.maximum(eta, z_bed)

    h_new = jnp.maximum(eta - z_bed, 0.0)

    river_tracer = _advect_tracer(
        river_tracer, h, h_new, u, v, h_e, h_n, dx, dy, dt, tracer_diffusion, river_mask, 1.0
    )
    ocean_tracer = _advect_tracer(
        ocean_tracer, h, h_new, u, v, h_e, h_n, dx, dy, dt, tracer_diffusion, ocean_inlet_mask, 1.0
    )

    return eta, u, v, river_tracer, ocean_tracer


@jit
def _apply_boundary_conditions_hydro(eta, u, v, z_bed, dam_mask, dam_level, river_mask, river_level, river_dh):
    """Apply hydrodynamic boundary conditions only."""
    eta = jnp.where(dam_mask, dam_level, eta)
    eta = jnp.where(river_mask, eta + river_dh, eta)
    eta = jnp.where(river_mask, jnp.maximum(eta, river_level), eta)
    return eta, u, v


@jit
def _apply_boundary_conditions_full(
    eta,
    u,
    v,
    river_tracer,
    ocean_tracer,
    z_bed,
    dam_mask,
    dam_level,
    river_mask,
    river_level,
    river_dh,
    ocean_inlet_mask,
):
    """Apply full boundary conditions including dual tracer injection."""
    eta = jnp.where(dam_mask, dam_level, eta)
    eta = jnp.where(river_mask, eta + river_dh, eta)
    eta = jnp.where(river_mask, jnp.maximum(eta, river_level), eta)

    h = jnp.maximum(eta - z_bed, 0.0)
    wet = h > 0.01
    river_tracer = jnp.where(river_mask & wet, 1.0, river_tracer)
    ocean_tracer = jnp.where(ocean_inlet_mask & wet, 1.0, ocean_tracer)

    return eta, u, v, river_tracer, ocean_tracer


class SimulationEngine:
    """GPU-accelerated shallow water simulation engine.

    Runs the JAX simulation and pushes frame data to a queue
    for parallel rendering.
    """

    def __init__(self, config: SimulationConfig, output_queue: Queue, log_fn=print):
        self.config = config
        self.output_queue = output_queue
        self.log = log_fn
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # These will be initialized in _setup()
        self.elev = None
        self.z_bed_np = None
        self.ny = 0
        self.nx = 0
        self.extent = (0, 0, 0, 0)
        self.gauges: list[dict] = []
        self.transformer = None

    def _load_mangrove_mask(self) -> "NDArray[np.bool_]":
        """Load mangrove zones from GeoJSON and rasterize to grid."""
        geojson_path = Path(__file__).parent.parent.parent.parent / "waitangi_mangroves.geojson"

        if not geojson_path.exists():
            self.log(f"Warning: Mangrove GeoJSON not found at {geojson_path}")
            return np.zeros((self.ny, self.nx), dtype=bool)

        self.log("Loading mangrove zones from GeoJSON...")

        with open(geojson_path) as f:
            data = json.load(f)

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)

        polygons = []
        for feature in data["features"]:
            geom = shape(feature["geometry"])
            if geom.geom_type == "Polygon":
                coords = list(geom.exterior.coords)
                transformed = [transformer.transform(lon, lat) for lon, lat in coords]
                polygons.append(Polygon(transformed))
            elif geom.geom_type == "MultiPolygon":
                for poly in geom.geoms:
                    coords = list(poly.exterior.coords)
                    transformed = [transformer.transform(lon, lat) for lon, lat in coords]
                    polygons.append(Polygon(transformed))

        if not polygons:
            self.log("Warning: No valid polygons found in mangrove GeoJSON")
            return np.zeros((self.ny, self.nx), dtype=bool)

        combined = unary_union(polygons)

        xmin, xmax, ymin, ymax = self.extent
        dx = (xmax - xmin) / self.nx
        dy = (ymax - ymin) / self.ny

        mask = np.zeros((self.ny, self.nx), dtype=bool)

        for i in range(self.ny):
            for j in range(self.nx):
                x = xmin + (j + 0.5) * dx
                y = ymax - (i + 0.5) * dy
                pt = Point(x, y)
                if combined.contains(pt):
                    mask[i, j] = True

        mangrove_cells = np.sum(mask)
        mangrove_area = mangrove_cells * dx * dy / 1e6
        self.log(f"Mangrove mask: {mangrove_cells} cells ({mangrove_area:.2f} km²)")

        return mask

    def _setup(self):
        """Initialize grid, load data, create masks."""
        self.log(f"\nJAX backend: {jax.default_backend()}")
        self.log(f"JAX devices: {jax.devices()}")

        self.log("\nLoading elevation data...")
        self.elev = fetch_waitangi_elevation()

        self.z_bed_np = self.elev.data[:: self.config.downsample, :: self.config.downsample].copy()
        self.ny, self.nx = self.z_bed_np.shape
        self.config.dx = self.config.dy = abs(self.elev.transform.a) * self.config.downsample

        self.log(f"Grid: {self.ny} x {self.nx} cells, {self.config.dx:.0f}m resolution")

        self.extent = (
            self.elev.bounds[0],
            self.elev.bounds[2],
            self.elev.bounds[1],
            self.elev.bounds[3],
        )

        # Set up domain boundaries
        self.z_bed_np[self.config.wall_north_row :, self.config.wall_col + 1 :] = 100.0
        self.z_bed_np[:8, :] = 20.0
        self.z_bed_np[-8:, :] = 20.0

        # Wall mask
        self.wall_mask_np = np.zeros((self.ny, self.nx), dtype=bool)
        self.wall_mask_np[self.config.wall_north_row :, self.config.wall_col] = True
        self.wall_mask_np = self.wall_mask_np & (self.z_bed_np < 10.0)
        self.z_bed_np[self.wall_mask_np] = np.minimum(
            self.z_bed_np[self.wall_mask_np], self.config.low_tide - 1.0
        )

        # River mask
        yy, xx = np.ogrid[: self.ny, : self.nx]
        self.river_mask_np = (
            (yy - self.config.falls_row) ** 2 + (xx - self.config.falls_col) ** 2
        ) <= self.config.river_radius**2

        # Ocean inlet mask - includes wall plus all domain boundaries where ocean water enters
        # This ensures any water entering from the edges gets tagged with ocean tracer
        self.ocean_inlet_mask_np = self.wall_mask_np.copy()
        # Add east edge (last column)
        self.ocean_inlet_mask_np[:, -1] = True
        # Add south edge (last row)
        self.ocean_inlet_mask_np[-1, :] = True
        # Add north edge (first row) - but only east of the river
        self.ocean_inlet_mask_np[0, self.config.falls_col + 10 :] = True

        self.log(f"Ocean inlet: {np.sum(self.ocean_inlet_mask_np)} cells (wall + domain edges)")
        self.log(
            f"River source: {np.sum(self.river_mask_np)} cells at row {self.config.falls_row}, col {self.config.falls_col}"
        )

        # Flow gauges
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
        self.gauges = [
            {"name": "Bridge", "row": 132, "col": 343, "lat": -35.271960, "lon": 174.079610},
            {"name": "River Mouth", "row": 185, "col": 170, "lat": -35.275924, "lon": 174.064415},
            {"name": "After Falls", "row": 217, "col": 52, "lat": -35.278341, "lon": 174.054052},
        ]
        for g in self.gauges:
            g["x"], g["y"] = self.transformer.transform(g["lon"], g["lat"])

        # Initialize kayak
        boat_ramp_x, boat_ramp_y = self.transformer.transform(BOAT_RAMP_LON, BOAT_RAMP_LAT)
        self.kayak_x = boat_ramp_x - 5.0
        self.kayak_y = boat_ramp_y - 5.0
        self.kayak_heading = 225.0
        self.log(f"Kayak start: ({self.kayak_x:.0f}, {self.kayak_y:.0f})")

        # Load mangrove mask and create Manning field
        mangrove_mask = self._load_mangrove_mask()
        self.manning_np = np.full_like(self.z_bed_np, self.config.manning_open_water)
        self.manning_np[mangrove_mask] = self.config.manning_mangrove
        self.log(
            f"Manning's n: {self.config.manning_open_water} (open) / {self.config.manning_mangrove} (mangrove)"
        )

        # Calculate timestep
        H_max = self.config.high_tide + 3.0
        self.dt = 0.2 * min(self.config.dx, self.config.dy) / np.sqrt(self.config.g * H_max)
        self.log(f"Timestep: {self.dt:.3f}s")

    def _run_simulation(self):
        """Main simulation loop - runs in a separate thread."""
        self._setup()

        cfg = self.config
        dx, dy, dt = cfg.dx, cfg.dy, self.dt

        # Convert to JAX arrays
        z_bed = jnp.array(self.z_bed_np)
        wall_mask = jnp.array(self.wall_mask_np)
        manning_field = jnp.array(self.manning_np)
        ocean_inlet_mask = jnp.array(self.ocean_inlet_mask_np)
        params = (dx, dy, dt, cfg.g, cfg.max_vel)

        # River mask: only inject tracer if river_flow > 0
        if cfg.river_flow > 0:
            river_mask = jnp.array(self.river_mask_np)
        else:
            river_mask = jnp.zeros((self.ny, self.nx), dtype=bool)
            self.log("River flow = 0: disabling river tracer injection")

        # River inflow per timestep
        river_area = np.sum(self.river_mask_np) * dx * dy
        river_dh = cfg.river_flow * dt / river_area if river_area > 0 else 0.0

        # Initialize water level
        # If skip_equilibrium: start completely dry (eta = z_bed, so h = 0 everywhere)
        # Otherwise: start at low tide and let it equilibrate
        if cfg.skip_equilibrium:
            self.log("\nStarting dry (skip_equilibrium)...")
            eta_np = self.z_bed_np.copy()
        else:
            init_level = cfg.low_tide
            self.log(f"\nInitializing at low tide ({init_level}m)...")
            eta_np = np.where(self.z_bed_np < init_level, init_level, self.z_bed_np)
            eta_np[self.wall_mask_np] = init_level

        eta = jnp.array(eta_np)
        u = jnp.zeros((self.ny, self.nx))
        v = jnp.zeros((self.ny, self.nx))
        river_tracer = jnp.zeros((self.ny, self.nx))
        ocean_tracer = jnp.zeros((self.ny, self.nx))

        empty_mask = jnp.zeros((self.ny, self.nx), dtype=bool)

        # Warm up JIT
        self.log("Warming up JIT...")
        eta, u, v, river_tracer, ocean_tracer = _simulation_step(
            eta, u, v, river_tracer, ocean_tracer, z_bed, manning_field, params, empty_mask, empty_mask, cfg.tracer_diffusion
        )
        eta, u, v = _apply_boundary_conditions_hydro(
            eta, u, v, z_bed, wall_mask, cfg.low_tide, river_mask, cfg.low_tide + 1.0, river_dh
        )
        jax.block_until_ready(eta)

        # Equilibration (can be skipped)
        if cfg.skip_equilibrium:
            self.log("Skipping equilibration phase...")
        else:
            self.log("Equilibrating at low tide...")
            for _ in range(2000):
                eta, u, v, river_tracer, ocean_tracer = _simulation_step(
                    eta, u, v, river_tracer, ocean_tracer, z_bed, manning_field, params, empty_mask, empty_mask, cfg.tracer_diffusion
                )
                eta, u, v = _apply_boundary_conditions_hydro(
                    eta, u, v, z_bed, wall_mask, cfg.low_tide, river_mask, cfg.low_tide + 1.0, river_dh
                )

        river_tracer = jnp.zeros((self.ny, self.nx))
        ocean_tracer = jnp.zeros((self.ny, self.nx))

        # Simulation timing
        if cfg.duration_hours is not None:
            duration = cfg.duration_hours * 3600
        else:
            duration = cfg.tide_period
        n_steps = int(duration / dt)
        steps_per_output = int(cfg.output_interval / dt)
        n_frames = n_steps // steps_per_output + 1

        self.log(f"\nRunning tidal cycle...")
        self.log(f"Duration: {duration/3600:.1f} hours ({n_steps} steps)")
        self.log(f"Output every {cfg.output_interval}s ({n_frames} frames)")

        frame_count = 0

        for step in range(n_steps):
            if self._stop_event.is_set():
                break

            t = step * dt
            if cfg.fixed_tide is not None:
                tide_level = cfg.fixed_tide
            else:
                tide_level = cfg.mean_tide + cfg.tide_amplitude * np.sin(2 * np.pi * t / cfg.tide_period - np.pi / 2)

            # Simulation step
            eta, u, v, river_tracer, ocean_tracer = _simulation_step(
                eta, u, v, river_tracer, ocean_tracer, z_bed, manning_field, params, river_mask, ocean_inlet_mask, cfg.tracer_diffusion
            )
            eta, u, v, river_tracer, ocean_tracer = _apply_boundary_conditions_full(
                eta, u, v, river_tracer, ocean_tracer, z_bed, wall_mask, tide_level, river_mask, tide_level + 1.0, river_dh, ocean_inlet_mask
            )

            # Advect kayak
            u_np = np.array(u)
            v_np = np.array(v)
            u_vel, v_vel = _interpolate_velocity(self.kayak_x, self.kayak_y, u_np, v_np, self.extent, self.ny, self.nx)
            self.kayak_x += u_vel * dt
            self.kayak_y += v_vel * dt
            if u_vel**2 + v_vel**2 > 0.001:
                self.kayak_heading = (90 - np.degrees(np.arctan2(v_vel, u_vel))) % 360

            # Output frame
            if step % steps_per_output == 0:
                jax.block_until_ready(eta)

                eta_np = np.array(eta)
                h = np.maximum(eta_np - self.z_bed_np, 0)
                river_tracer_np = np.array(river_tracer)
                ocean_tracer_np = np.array(ocean_tracer)

                wet_area_km2 = np.sum(h > 0.01) * dx * dy / 1e6

                # Collect gauge data
                gauge_data = []
                for g in self.gauges:
                    r, c = g["row"], g["col"]
                    gh = h[r, c]
                    gu = u_np[r, c]
                    gv = v_np[r, c]
                    speed = np.sqrt(gu**2 + gv**2)
                    flow = speed * gh * dx
                    direction = _velocity_to_compass(gu, gv)
                    gauge_data.append(
                        GaugeData(
                            name=g["name"],
                            depth=gh,
                            speed=speed,
                            flow=flow,
                            direction=direction,
                            x=g["x"],
                            y=g["y"],
                        )
                    )

                frame_data = FrameData(
                    frame_number=frame_count,
                    total_frames=n_frames,
                    simulation_time=t,
                    tide_level=tide_level,
                    h=h.copy(),
                    u=u_np.copy(),
                    v=v_np.copy(),
                    river_tracer=river_tracer_np.copy(),
                    ocean_tracer=ocean_tracer_np.copy(),
                    z_bed=self.z_bed_np,
                    wet_area_km2=wet_area_km2,
                    gauges=gauge_data,
                    kayak=KayakState(x=self.kayak_x, y=self.kayak_y, heading=self.kayak_heading),
                    river_flow=cfg.river_flow,
                )

                self.output_queue.put(frame_data)
                frame_count += 1

                progress = (step + 1) / n_steps * 100
                self.log(
                    f"  {progress:5.1f}% | t={t/3600:.2f}h | tide={tide_level:+.2f}m | "
                    f"area={wet_area_km2:.2f}km² | frame {frame_count}/{n_frames}"
                )

        # Signal end of stream
        self.output_queue.put(END_OF_STREAM)
        self.log("\nSimulation complete.")

    def start(self):
        """Start the simulation in a background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_simulation, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the simulation gracefully."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def join(self):
        """Wait for simulation to complete."""
        if self._thread:
            self._thread.join()

    @property
    def grid_extent(self) -> tuple[float, float, float, float]:
        """Return grid extent (xmin, xmax, ymin, ymax)."""
        return self.extent

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Return grid shape (ny, nx)."""
        return self.ny, self.nx

    @property
    def downsample_factor(self) -> int:
        """Return downsample factor."""
        return self.config.downsample

    @property
    def elevation_data(self):
        """Return raw elevation data."""
        return self.elev
