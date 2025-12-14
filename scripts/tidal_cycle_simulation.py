#!/usr/bin/env python3
"""Tidal cycle simulation using JAX for GPU acceleration.

Dam spillway approach:
- Artificial wall (dam) east of bridge, height follows tide level
- Rising tide: dam rises, water fills basin
- Falling tide: dam lowers, water spills out over dam

Features:
- Spatially-varying Manning friction (higher in mangroves)
- Dual tracer system: river (green) and ocean (pink)
- Configurable river flow for heavy rainfall scenarios
"""

import json
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import jit
from matplotlib.colors import LinearSegmentedColormap
from pyproj import Transformer
from scipy import ndimage
from shapely.geometry import shape, Point

from waitangi.data.elevation import fetch_waitangi_elevation


def log(msg: str) -> None:
    """Print with immediate flush for visibility."""
    print(msg, flush=True)


# Use 32-bit precision (required for Metal GPU backend)
jax.config.update("jax_enable_x64", False)


# Manning's n coefficients
MANNING_OPEN_WATER = 0.035  # Open channels
MANNING_MANGROVE = 0.12     # Dense mangrove vegetation


def load_mangrove_mask(ny: int, nx: int, extent: tuple, downsample: int) -> np.ndarray:
    """Load mangrove zones from GeoJSON and rasterize to grid.

    Args:
        ny, nx: Grid dimensions
        extent: (xmin, xmax, ymin, ymax) in NZTM coordinates
        downsample: Grid downsample factor

    Returns:
        Boolean mask where True = mangrove zone
    """
    geojson_path = Path(__file__).parent.parent / "waitangi_mangroves.geojson"

    if not geojson_path.exists():
        log(f"Warning: Mangrove GeoJSON not found at {geojson_path}")
        return np.zeros((ny, nx), dtype=bool)

    log("Loading mangrove zones from GeoJSON...")

    with open(geojson_path) as f:
        data = json.load(f)

    # Transform from WGS84 to NZTM
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)

    # Extract and transform all polygons
    polygons = []
    for feature in data["features"]:
        geom = shape(feature["geometry"])
        # Transform coordinates
        if geom.geom_type == "Polygon":
            coords = list(geom.exterior.coords)
            transformed = [transformer.transform(lon, lat) for lon, lat in coords]
            from shapely.geometry import Polygon
            polygons.append(Polygon(transformed))
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                coords = list(poly.exterior.coords)
                transformed = [transformer.transform(lon, lat) for lon, lat in coords]
                from shapely.geometry import Polygon
                polygons.append(Polygon(transformed))

    if not polygons:
        log("Warning: No valid polygons found in mangrove GeoJSON")
        return np.zeros((ny, nx), dtype=bool)

    # Combine all polygons
    from shapely.ops import unary_union
    combined = unary_union(polygons)

    # Rasterize to grid
    xmin, xmax, ymin, ymax = extent
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    mask = np.zeros((ny, nx), dtype=bool)

    # Check each grid cell center
    for i in range(ny):
        for j in range(nx):
            # Grid uses upper-left origin, so y decreases with row
            x = xmin + (j + 0.5) * dx
            y = ymax - (i + 0.5) * dy
            pt = Point(x, y)
            if combined.contains(pt):
                mask[i, j] = True

    mangrove_cells = np.sum(mask)
    mangrove_area = mangrove_cells * dx * dy / 1e6
    log(f"Mangrove mask: {mangrove_cells} cells ({mangrove_area:.2f} km²)")

    return mask


def create_manning_field(z_bed: np.ndarray, mangrove_mask: np.ndarray) -> np.ndarray:
    """Create spatially-varying Manning's n field.

    Args:
        z_bed: Bed elevation array
        mangrove_mask: Boolean mask of mangrove zones

    Returns:
        Manning's n coefficient at each grid cell
    """
    manning = np.full_like(z_bed, MANNING_OPEN_WATER)
    manning[mangrove_mask] = MANNING_MANGROVE
    return manning


def create_dual_tracer_colormap():
    """Create colormap for dual tracer visualization.

    Tracer values:
    - 0.0 = pure ocean (pink)
    - 0.5 = neutral/untainted water (blue)
    - 1.0 = pure river (green)
    """
    colors = [
        (0.0, (0.9, 0.2, 0.6, 0.9)),   # Bright pink (pure ocean)
        (0.25, (0.7, 0.4, 0.8, 0.85)), # Light pink-purple
        (0.5, (0.2, 0.5, 0.9, 0.85)),  # Blue (neutral/untainted)
        (0.75, (0.4, 0.8, 0.4, 0.85)), # Light green
        (1.0, (0.1, 0.95, 0.1, 0.9)),  # Bright green (pure river)
    ]
    return LinearSegmentedColormap.from_list("ocean_river", colors)


def velocity_to_compass(u: float, v: float) -> str:
    """Convert u,v velocity components to compass direction.

    Note: In our grid, u is east-west (positive = east) and v is north-south (positive = north).
    """
    speed = np.sqrt(u**2 + v**2)
    if speed < 0.001:
        return "--"

    # Calculate angle in degrees (0 = East, 90 = North)
    angle = np.degrees(np.arctan2(v, u))
    # Convert to compass bearing (0 = North, 90 = East)
    bearing = (90 - angle) % 360

    # 8-point compass
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((bearing + 22.5) / 45) % 8
    return directions[idx]


@jit
def simulation_step(eta, u, v, tracer, z_bed, manning_field, params, tracer_diffusion: float = 5.0):
    """Single simulation timestep - JIT compiled for GPU.

    Args:
        eta: Water surface elevation
        u, v: Velocity components
        tracer: River tracer concentration (0=ocean, 1=river)
        z_bed: Bed elevation
        manning_field: Spatially-varying Manning's n coefficient
        params: (dx, dy, dt, g, max_vel)
        tracer_diffusion: Diffusion coefficient for tracer (0 = no diffusion, permanent dye)

    Returns:
        Updated eta, u, v, tracer
    """
    dx, dy, dt, g, max_vel = params

    # Depth
    h = jnp.maximum(eta - z_bed, 0.0)

    # Surface gradient
    deta_dx = jnp.zeros_like(eta)
    deta_dx = deta_dx.at[:-1, :].set((eta[1:, :] - eta[:-1, :]) / dx)

    deta_dy = jnp.zeros_like(eta)
    deta_dy = deta_dy.at[:, :-1].set((eta[:, 1:] - eta[:, :-1]) / dy)

    # Velocity update
    u = u.at[:-1, :].add(-g * dt * deta_dx[:-1, :])
    v = v.at[:, :-1].add(-g * dt * deta_dy[:, :-1])

    # Friction (implicit) - spatially varying Manning's n
    h_at_u = jnp.maximum((h[:-1, :] + h[1:, :]) / 2, 0.01)
    h_at_v = jnp.maximum((h[:, :-1] + h[:, 1:]) / 2, 0.01)

    # Average Manning's n at cell faces
    manning_at_u = (manning_field[:-1, :] + manning_field[1:, :]) / 2
    manning_at_v = (manning_field[:, :-1] + manning_field[:, 1:]) / 2

    Cf_u = g * manning_at_u**2 / jnp.power(h_at_u, 1/3)
    Cf_v = g * manning_at_v**2 / jnp.power(h_at_v, 1/3)

    friction_u = 1.0 / (1.0 + dt * Cf_u * jnp.abs(u[:-1, :]) / h_at_u)
    friction_v = 1.0 / (1.0 + dt * Cf_v * jnp.abs(v[:, :-1]) / h_at_v)

    u = u.at[:-1, :].multiply(friction_u)
    v = v.at[:, :-1].multiply(friction_v)

    # Velocity limiting
    u = jnp.clip(u, -max_vel, max_vel)
    v = jnp.clip(v, -max_vel, max_vel)

    # Wall boundaries
    u = u.at[-1, :].set(0.0)
    u = u.at[0, :].set(0.0)
    v = v.at[:, -1].set(0.0)
    v = v.at[:, 0].set(0.0)

    # Upwind depth for mass flux
    h_e = jnp.where(u[:-1, :] > 0, h[:-1, :], h[1:, :])
    h_n = jnp.where(v[:, :-1] > 0, h[:, :-1], h[:, 1:])

    # Mass flux
    flux_e = u[:-1, :] * h_e
    flux_n = v[:, :-1] * h_n

    # Divergence
    div = jnp.zeros_like(eta)
    div = div.at[:-1, :].add(flux_e / dx)
    div = div.at[1:, :].add(-flux_e / dx)
    div = div.at[:, :-1].add(flux_n / dy)
    div = div.at[:, 1:].add(-flux_n / dy)

    # Update eta
    eta = eta - dt * div
    eta = jnp.maximum(eta, z_bed)

    # Tracer advection (upwind)
    h_new = jnp.maximum(eta - z_bed, 0.0)

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
    wet_new = h_new > 0.01
    # Dry cells keep neutral value (0.5) - prevents pink leaking from boundaries
    tracer = jnp.where(wet_new, (tracer * h - dt * tracer_div) / h_safe, 0.5)
    tracer = jnp.clip(tracer, 0.0, 1.0)

    # Tracer diffusion (optional - set to 0 for permanent dye tracking)
    # Pad with 0.5 (neutral) to prevent boundary artifacts
    tracer_padded = jnp.pad(tracer, 1, mode='constant', constant_values=0.5)
    lap = (
        tracer_padded[:-2, 1:-1] + tracer_padded[2:, 1:-1] +
        tracer_padded[1:-1, :-2] + tracer_padded[1:-1, 2:] - 4 * tracer
    ) / (dx * dy)
    tracer = tracer + tracer_diffusion * dt * lap
    tracer = jnp.clip(tracer, 0.0, 1.0)
    # Ensure dry cells stay neutral
    tracer = jnp.where(wet_new, tracer, 0.5)

    return eta, u, v, tracer


@jit
def apply_boundary_conditions_hydro(eta, u, v, z_bed, dam_mask, dam_level,
                                     river_mask, river_level, river_dh):
    """Apply hydrodynamic boundary conditions only (no tracer injection).

    Dam acts as spillway:
    - If eta > dam_level: water spills out (set eta = dam_level)
    - If eta < dam_level: water flows in (set eta = dam_level)
    This maintains equilibrium at exactly dam_level.
    """
    # Dam spillway: clamp water level to dam height
    eta = jnp.where(dam_mask, dam_level, eta)

    # River source: add water
    eta = jnp.where(river_mask, eta + river_dh, eta)
    eta = jnp.where(river_mask, jnp.maximum(eta, river_level), eta)

    return eta, u, v


@jit
def apply_boundary_conditions_full(eta, u, v, tracer, z_bed, dam_mask, dam_level,
                                    river_mask, river_level, river_dh,
                                    ocean_inlet_mask):
    """Apply full boundary conditions including tracer injection.

    Dam acts as spillway (controls water level across entire wall).
    Ocean tracer injected only at ocean_inlet_mask (small section).
    River source adds water + injects river tracer (1).
    """
    # Dam spillway: clamp water level to dam height
    eta = jnp.where(dam_mask, dam_level, eta)

    # River source: add water
    eta = jnp.where(river_mask, eta + river_dh, eta)
    eta = jnp.where(river_mask, jnp.maximum(eta, river_level), eta)

    # Tracer sources - use ocean_inlet_mask for tracer (not entire dam_mask)
    h = jnp.maximum(eta - z_bed, 0.0)
    wet = h > 0.01
    tracer = jnp.where(river_mask & wet, 1.0, tracer)        # River = green
    tracer = jnp.where(ocean_inlet_mask & wet, 0.0, tracer)  # Ocean inlet = pink

    return eta, u, v, tracer


def run_equilibrium_test():
    """Test: dump water at high tide, let it reach equilibrium."""
    log("=" * 60)
    log("JAX Equilibrium Test - Instant Fill")
    log("=" * 60)

    # Check JAX backend
    log(f"\nJAX backend: {jax.default_backend()}")
    log(f"JAX devices: {jax.devices()}")

    # Load elevation
    log("\nLoading elevation data...")
    elev = fetch_waitangi_elevation()

    downsample = 8
    z_bed_np = elev.data[::downsample, ::downsample].copy()
    ny, nx = z_bed_np.shape
    dx = dy = abs(elev.transform.a) * downsample

    log(f"Grid: {ny} x {nx} cells, {dx:.0f}m resolution")

    # Tide levels
    high_tide = 1.1
    dam_col = 350

    # Create closed basin
    z_bed_np[:, dam_col:] = high_tide  # Dam at high tide level
    z_bed_np[:8, :] = 20.0   # North wall
    z_bed_np[-8:, :] = 20.0  # South wall

    # Dam mask - the spillway cells at the dam wall
    dam_col_start = dam_col - 3
    dam_mask_np = np.zeros((ny, nx), dtype=bool)
    dam_mask_np[:, dam_col_start:dam_col_start+2] = True

    # River source at Haruru Falls
    falls_row, falls_col = 214, 24
    river_radius = 3
    yy, xx = np.ogrid[:ny, :nx]
    river_mask_np = ((yy - falls_row)**2 + (xx - falls_col)**2) <= river_radius**2

    # Physical parameters
    g = 9.81
    manning_n = 0.035
    max_vel = 3.0
    river_flow = 1.0  # m³/s

    # CFL timestep
    H_max = high_tide + 3.0
    dt = 0.2 * min(dx, dy) / np.sqrt(g * H_max)
    log(f"Timestep: {dt:.3f}s")

    # Convert to JAX arrays
    z_bed = jnp.array(z_bed_np)
    dam_mask = jnp.array(dam_mask_np)
    river_mask = jnp.array(river_mask_np)
    params = (dx, dy, dt, g, manning_n, max_vel)

    # River inflow per timestep (disabled for equilibrium test)
    river_area = np.sum(river_mask_np) * dx * dy
    river_dh = 0.0  # No river for this test

    # INSTANT FILL: Start with water at high tide everywhere below that level
    log(f"\nInstant fill to high tide level: {high_tide}m")
    eta_np = np.maximum(z_bed_np, high_tide)  # Water surface at high tide
    eta_np = np.where(z_bed_np < high_tide, high_tide, z_bed_np)  # Only where ground is below

    eta = jnp.array(eta_np)
    u = jnp.zeros((ny, nx))
    v = jnp.zeros((ny, nx))
    tracer = jnp.zeros((ny, nx))

    # Initial state
    h_init = np.maximum(eta_np - z_bed_np, 0)
    init_area = np.sum(h_init > 0.01) * dx * dy / 1e6
    init_volume = np.sum(h_init) * dx * dy / 1e6
    log(f"Initial wet area: {init_area:.2f} km²")
    log(f"Initial volume: {init_volume:.3f} million m³")

    # Warm up JIT
    log("\nWarming up JIT compilation...")
    start = time.time()
    eta, u, v, tracer = simulation_step(eta, u, v, tracer, z_bed, params)
    eta, u, v, tracer = apply_boundary_conditions_full(
        eta, u, v, tracer, z_bed, dam_mask, high_tide,
        river_mask, high_tide + 1.0, river_dh
    )
    jax.block_until_ready(eta)
    log(f"JIT compile time: {time.time() - start:.2f}s")

    # Run to equilibrium
    n_steps = 20000
    print_interval = 2000

    log(f"\nRunning {n_steps} steps to equilibrium...")
    start = time.time()

    for step in range(n_steps):
        eta, u, v, tracer = simulation_step(eta, u, v, tracer, z_bed, params)
        eta, u, v, tracer = apply_boundary_conditions_full(
            eta, u, v, tracer, z_bed, dam_mask, high_tide,
            river_mask, high_tide + 1.0, river_dh
        )

        if (step + 1) % print_interval == 0:
            jax.block_until_ready(eta)
            eta_np = np.array(eta)
            h = np.maximum(eta_np - np.array(z_bed), 0)
            wet_area = np.sum(h > 0.01) * dx * dy / 1e6
            max_h = np.max(h)
            max_speed = np.sqrt(np.max(np.array(u)**2 + np.array(v)**2))
            elapsed = time.time() - start
            steps_per_sec = (step + 1) / elapsed
            log(f"  Step {step+1}: area={wet_area:.2f}km², max_h={max_h:.2f}m, "
                  f"max_v={max_speed:.2f}m/s, {steps_per_sec:.0f} steps/s")

    elapsed = time.time() - start
    log(f"\nCompleted in {elapsed:.1f}s ({n_steps/elapsed:.0f} steps/s)")

    # Final state
    eta_np = np.array(eta)
    h_final = np.maximum(eta_np - np.array(z_bed), 0)
    final_area = np.sum(h_final > 0.01) * dx * dy / 1e6
    final_volume = np.sum(h_final) * dx * dy / 1e6

    log(f"\n{'='*60}")
    log("Results")
    log(f"{'='*60}")
    log(f"Final wet area: {final_area:.2f} km²")
    log(f"Final volume: {final_volume:.3f} million m³")
    log(f"Max depth: {np.max(h_final):.2f}m")

    # Expected flood area (flood fill from dam)
    potentially_wet = np.array(z_bed) < high_tide
    struct = ndimage.generate_binary_structure(2, 1)
    flooded = dam_mask_np.copy()
    for _ in range(max(ny, nx)):
        expanded = ndimage.binary_dilation(flooded, structure=struct)
        new_flooded = expanded & potentially_wet
        if np.array_equal(new_flooded, flooded):
            break
        flooded = new_flooded
    expected_area = np.sum(flooded) * dx * dy / 1e6

    log(f"\nExpected flooded area: {expected_area:.2f} km²")
    log(f"Difference: {final_area - expected_area:.3f} km²")

    # Save comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    extent = [elev.bounds[0], elev.bounds[2], elev.bounds[1], elev.bounds[3]]

    axes[0].imshow(flooded, origin='upper', extent=extent, cmap='Blues')
    axes[0].set_title(f'Expected Flood at {high_tide}m')
    axes[0].set_xlabel('Easting (m)')
    axes[0].set_ylabel('Northing (m)')

    wet_final = h_final > 0.01
    axes[1].imshow(wet_final, origin='upper', extent=extent, cmap='Blues')
    axes[1].set_title(f'Simulated (after {n_steps} steps)')
    axes[1].set_xlabel('Easting (m)')

    overflow = wet_final & ~flooded
    underfill = flooded & ~wet_final
    diff = np.zeros_like(wet_final, dtype=int)
    diff[overflow] = 1
    diff[underfill] = -1
    axes[2].imshow(diff, origin='upper', extent=extent, cmap='RdYlBu', vmin=-1, vmax=1)
    axes[2].set_title('Difference (red=overflow, blue=underfill)')
    axes[2].set_xlabel('Easting (m)')

    plt.tight_layout()
    plt.savefig('jax_equilibrium_test.png', dpi=150)
    log(f"\nSaved plot to jax_equilibrium_test.png")

    return final_area, expected_area


def run_tidal_cycle(river_flow: float = 1.0, tracer_diffusion: float = 0.5, duration_hours: float | None = None):
    """Full tidal cycle simulation with wall boundary.

    The wall is a 1-cell-wide barrier at the eastern edge of the estuary.
    - Wall height = current tide level
    - Rising tide: wall rises, water flows in from "ocean"
    - Falling tide: wall lowers, water spills over and leaves simulation

    Args:
        river_flow: River discharge in m³/s (default 1.0, use 15.0 for heavy rain)
        tracer_diffusion: Diffusion coefficient for tracer (default 5.0, use 0 for permanent dye)
    """
    log("=" * 60)
    log("JAX Tidal Cycle Simulation")
    log("=" * 60)
    log(f"River flow: {river_flow} m³/s" + (" (HEAVY RAINFALL)" if river_flow > 5 else ""))
    log(f"Tracer diffusion: {tracer_diffusion}" + (" (PERMANENT DYE)" if tracer_diffusion == 0 else ""))

    log(f"\nJAX backend: {jax.default_backend()}")
    log(f"JAX devices: {jax.devices()}")

    # Load elevation
    log("\nLoading elevation data...")
    elev = fetch_waitangi_elevation()

    downsample = 8
    z_bed_np = elev.data[::downsample, ::downsample].copy()
    ny, nx = z_bed_np.shape
    dx = dy = abs(elev.transform.a) * downsample

    log(f"Grid: {ny} x {nx} cells, {dx:.0f}m resolution")

    # Tidal parameters
    low_tide = -0.5
    high_tide = 1.1
    mean_tide = (low_tide + high_tide) / 2
    tide_amplitude = (high_tide - low_tide) / 2
    tide_period = 12.42 * 3600  # M2 tide in seconds

    # Wall location - at longitude 174.083008 (-35.271313, 174.083008)
    # Calculated: NZTM x=1698501 -> column 382
    wall_col = 382

    # Set up the domain:
    # - Everything east of wall is "outside" (set bed very high so no water there)
    # - The wall itself will have eta clamped to tide level
    # - North/south boundaries are walls
    z_bed_np[:, wall_col+1:] = 100.0  # Outside domain - very high
    z_bed_np[:8, :] = 20.0   # North wall
    z_bed_np[-8:, :] = 20.0  # South wall

    # Wall mask - the single column where we enforce tide level
    wall_mask_np = np.zeros((ny, nx), dtype=bool)
    wall_mask_np[:, wall_col] = True
    # Only where there's actually water channel (not on land)
    wall_mask_np = wall_mask_np & (z_bed_np < 10.0)

    # River source at Haruru Falls - circular mask
    falls_row, falls_col = 214, 24
    river_radius = 3
    yy, xx = np.ogrid[:ny, :nx]
    river_mask_np = ((yy - falls_row)**2 + (xx - falls_col)**2) <= river_radius**2
    river_cells = np.sum(river_mask_np)

    # Ocean inlet mask - this is where we COLOR incoming ocean water pink
    # The wall boundary at column 382 is where water enters when tide rises
    # We need to color water AT THE WALL, not inside the domain
    # Use the wall mask itself as the ocean inlet - all water entering through the wall is ocean water
    ocean_inlet_mask_np = wall_mask_np.copy()
    ocean_inlet_cells = np.sum(ocean_inlet_mask_np)
    log(f"Ocean inlet: using wall mask ({ocean_inlet_cells} cells at column {wall_col})")
    log(f"River source: {river_cells} cells at row {falls_row}, col {falls_col}")

    # Virtual flow gauges - measure velocity and depth at key locations
    # Coordinates converted from WGS84 to grid indices
    gauges = [
        {"name": "Bridge", "row": 132, "col": 343, "lat": -35.271960, "lon": 174.079610},
        {"name": "River Mouth", "row": 185, "col": 170, "lat": -35.275924, "lon": 174.064415},
        {"name": "After Falls", "row": 217, "col": 52, "lat": -35.278341, "lon": 174.054052},
    ]
    # Convert to NZTM for plotting markers
    gauge_transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
    for g in gauges:
        g["x"], g["y"] = gauge_transformer.transform(g["lon"], g["lat"])
    log(f"Flow gauges: {[g['name'] for g in gauges]}")

    log(f"Tide range: {low_tide}m to {high_tide}m")
    log(f"Wall at column {wall_col}")
    log(f"Wall cells: {np.sum(wall_mask_np)}")

    # Load mangrove zones and create spatially-varying Manning field
    extent = (elev.bounds[0], elev.bounds[2], elev.bounds[1], elev.bounds[3])
    mangrove_mask = load_mangrove_mask(ny, nx, extent, downsample)
    manning_np = create_manning_field(z_bed_np, mangrove_mask)
    log(f"Manning's n: {MANNING_OPEN_WATER} (open) / {MANNING_MANGROVE} (mangrove)")

    # Physical parameters
    g = 9.81
    max_vel = 3.0

    # CFL timestep
    H_max = high_tide + 3.0
    dt = 0.2 * min(dx, dy) / np.sqrt(g * H_max)
    log(f"Timestep: {dt:.3f}s")

    # Convert to JAX (these are FIXED for the whole simulation)
    z_bed = jnp.array(z_bed_np)
    wall_mask = jnp.array(wall_mask_np)
    river_mask = jnp.array(river_mask_np)
    manning_field = jnp.array(manning_np)
    ocean_inlet_mask = jnp.array(ocean_inlet_mask_np)
    params = (dx, dy, dt, g, max_vel)

    # River inflow per timestep
    river_area = np.sum(river_mask_np) * dx * dy
    river_dh = river_flow * dt / river_area if river_area > 0 else 0.0

    # Initialize at low tide - fill basin to low tide level
    log(f"\nInitializing at low tide ({low_tide}m)...")
    eta_np = np.where(z_bed_np < low_tide, low_tide, z_bed_np)
    eta_np[wall_mask_np] = low_tide  # Wall at tide level

    eta = jnp.array(eta_np)
    u = jnp.zeros((ny, nx))
    v = jnp.zeros((ny, nx))
    # Tracer: 0=ocean(pink), 0.5=neutral(blue), 1=river(green)
    # Start with neutral blue water everywhere
    tracer = jnp.full((ny, nx), 0.5)

    # Warm up JIT (no tracer injection during warmup)
    log("Warming up JIT...")
    eta, u, v, tracer = simulation_step(eta, u, v, tracer, z_bed, manning_field, params, tracer_diffusion)
    eta, u, v = apply_boundary_conditions_hydro(
        eta, u, v, z_bed, wall_mask, low_tide,
        river_mask, low_tide + 1.0, river_dh
    )
    jax.block_until_ready(eta)

    # Quick equilibration at low tide (NO tracer injection - keep water neutral blue)
    log("Equilibrating at low tide (no tracer injection)...")
    for _ in range(2000):
        eta, u, v, tracer = simulation_step(eta, u, v, tracer, z_bed, manning_field, params, tracer_diffusion)
        eta, u, v = apply_boundary_conditions_hydro(
            eta, u, v, z_bed, wall_mask, low_tide,
            river_mask, low_tide + 1.0, river_dh
        )

    # Reset tracer to neutral after equilibration
    tracer = jnp.full((ny, nx), 0.5)

    # Simulation parameters
    if duration_hours is not None:
        duration = duration_hours * 3600
    else:
        duration = tide_period  # One full cycle
    n_steps = int(duration / dt)
    output_interval = 300  # 5 minutes between frames
    steps_per_output = int(output_interval / dt)

    log(f"\nRunning tidal cycle...")
    log(f"Duration: {duration/3600:.1f} hours ({n_steps} steps)")
    log(f"Output every {output_interval}s ({n_steps // steps_per_output} frames)")

    # Storage
    frames = []
    frame_times = []

    start_time = time.time()

    for step in range(n_steps):
        t = step * dt

        # Current tide level (start at low tide, rising)
        tide_level = mean_tide + tide_amplitude * np.sin(2 * np.pi * t / tide_period - np.pi/2)

        # Simulation step with spatially-varying friction
        eta, u, v, tracer = simulation_step(eta, u, v, tracer, z_bed, manning_field, params, tracer_diffusion)

        # Apply boundary: wall cells are clamped to tide level
        # This is how water flows in (rising) or out (falling)
        # Use full version to inject tracers during main simulation
        eta, u, v, tracer = apply_boundary_conditions_full(
            eta, u, v, tracer, z_bed, wall_mask, tide_level,
            river_mask, tide_level + 1.0, river_dh,
            ocean_inlet_mask
        )

        # Save frame
        if step % steps_per_output == 0:
            jax.block_until_ready(eta)
            eta_np = np.array(eta)
            u_np = np.array(u)
            v_np = np.array(v)
            h = np.maximum(eta_np - z_bed_np, 0)

            # Collect gauge measurements
            gauge_data = []
            for g in gauges:
                r, c = g["row"], g["col"]
                gh = h[r, c]
                gu = u_np[r, c]
                gv = v_np[r, c]
                speed = np.sqrt(gu**2 + gv**2)
                # Flow rate Q = velocity * depth * width (estimate width as dx)
                flow = speed * gh * dx  # m³/s (approximate, assumes flow perpendicular to cell)
                direction = velocity_to_compass(gu, gv)
                gauge_data.append({
                    "name": g["name"],
                    "depth": gh,
                    "speed": speed,
                    "flow": flow,
                    "direction": direction,
                    "u": gu,
                    "v": gv,
                })

            frames.append({
                'h': h.copy(),
                'tracer': np.array(tracer).copy(),
                'tide': tide_level,
                'gauges': gauge_data,
            })
            frame_times.append(t)

            wet_area = np.sum(h > 0.01) * dx * dy / 1e6
            elapsed = time.time() - start_time
            progress = (step + 1) / n_steps * 100
            log(f"  {progress:5.1f}% | t={t/3600:.2f}h | tide={tide_level:+.2f}m | "
                  f"area={wet_area:.2f}km² | {elapsed:.0f}s elapsed")
            # Log gauge data
            gauge_str = " | ".join([f"{g['name']}: h={g['depth']:.2f}m v={g['speed']:.3f}m/s" for g in gauge_data])
            log(f"    Gauges: {gauge_str}")

    elapsed = time.time() - start_time
    log(f"\nSimulation complete in {elapsed:.1f}s ({n_steps/elapsed:.0f} steps/s)")
    log(f"Generated {len(frames)} frames")

    # Render frames
    log("\n--- Rendering frames ---")
    output_dir = Path("tidal_cycle_frames")
    output_dir.mkdir(exist_ok=True)

    for old in output_dir.glob("frame_*.png"):
        old.unlink()

    extent = [elev.bounds[0], elev.bounds[2], elev.bounds[1], elev.bounds[3]]
    cmap = create_dual_tracer_colormap()

    import contextily as ctx
    from scipy.ndimage import zoom as scipy_zoom

    for i, (frame, t) in enumerate(zip(frames, frame_times)):
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        try:
            ctx.add_basemap(ax, crs="EPSG:2193", source=ctx.providers.Esri.WorldImagery, zoom=15)
        except Exception as e:
            log(f"  Warning: basemap failed: {e}")

        # Upsample for display
        h_full = scipy_zoom(frame['h'], downsample, order=1)
        tracer_full = scipy_zoom(frame['tracer'], downsample, order=1)
        h_full = h_full[:elev.data.shape[0], :elev.data.shape[1]]
        tracer_full = tracer_full[:elev.data.shape[0], :elev.data.shape[1]]

        wet = h_full > 0.05
        display = np.where(wet, tracer_full, np.nan)

        im = ax.imshow(display, cmap=cmap, origin='upper', extent=extent,
                       vmin=0, vmax=1, alpha=0.85)

        # Plot gauge markers
        marker_colors = ['yellow', 'cyan', 'lime']
        for gi, g in enumerate(gauges):
            ax.plot(g["x"], g["y"], 'o', color=marker_colors[gi], markersize=10,
                    markeredgecolor='black', markeredgewidth=1.5)
            ax.annotate(g["name"], (g["x"], g["y"]), xytext=(5, 5),
                        textcoords='offset points', fontsize=8, fontweight='bold',
                        color='white', path_effects=[
                            plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='black')
                        ])

        cbar = plt.colorbar(im, ax=ax, shrink=0.4, pad=0.02)
        cbar.set_label('Water Source', fontsize=10)
        cbar.set_ticks([0, 0.5, 1.0])
        cbar.set_ticklabels(['Ocean (pink)', 'Neutral (blue)', 'River (green)'])

        wet_area_km2 = np.sum(frame['h'] > 0.01) * dx * dy / 1e6
        hours = int(t // 3600)
        mins = int((t % 3600) // 60)

        # Build gauge info text
        gauge_text = "Flow Gauges:\n"
        for gi, gd in enumerate(frame['gauges']):
            gauge_text += f"{gd['name']:12s} h={gd['depth']:.2f}m  v={gd['speed']:.2f}m/s {gd['direction']:>2s}  Q={gd['flow']:.1f}m³/s\n"

        ax.set_title(
            f"Waitangi Estuary - Tidal Simulation\n"
            f"Time: {hours}h {mins:02d}m | Tide: {frame['tide']:+.2f}m | River: {river_flow:.1f} m³/s\n"
            f"Flooded: {wet_area_km2:.2f} km² | Frame {i+1}/{len(frames)}",
            fontsize=12, fontweight='bold'
        )
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")

        # Add gauge data text box (bottom right, left-aligned text)
        ax.text(0.98, 0.02, gauge_text.strip(), transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                fontfamily='monospace', multialignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        plt.tight_layout()
        plt.savefig(output_dir / f"frame_{i:04d}.png", dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()

        if (i + 1) % 10 == 0:
            log(f"  Rendered {i+1}/{len(frames)} frames")

    # Create video
    log("\n--- Creating video ---")
    video_path = "tidal_cycle.mp4"
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", "10",
            "-i", str(output_dir / "frame_%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-crf", "18",
            video_path,
        ], check=True, capture_output=True, text=True)
        log(f"Video saved to: {video_path}")
    except subprocess.CalledProcessError as e:
        log(f"ffmpeg error: {e.stderr}")
    except FileNotFoundError:
        log("ffmpeg not found - frames saved but video not created")

    log(f"\nDone! Frames saved to: {output_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Waitangi Estuary Tidal Simulation")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "test"],
                        help="Command: 'run' for tidal cycle, 'test' for equilibrium test")
    parser.add_argument("--river-flow", type=float, default=1.0,
                        help="River flow rate in m³/s (default: 1.0, use 15.0 for heavy rain)")
    parser.add_argument("--no-diffusion", action="store_true",
                        help="Disable tracer diffusion (permanent dye tracking)")
    parser.add_argument("--diffusion", type=float, default=0.5,
                        help="Tracer diffusion coefficient (default: 0.5, realistic for estuarine turbulence)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Simulation duration in hours (default: full tidal cycle ~12.4h)")

    args = parser.parse_args()

    tracer_diffusion = 0.0 if args.no_diffusion else args.diffusion

    if args.command == "test":
        run_equilibrium_test()
    else:
        run_tidal_cycle(river_flow=args.river_flow, tracer_diffusion=tracer_diffusion,
                        duration_hours=args.duration)
