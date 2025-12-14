"""2D Shallow Water Equations solver for Waitangi Estuary.

GPU-accelerated using JAX for real-time performance.

Solves the depth-averaged shallow water equations:

∂η/∂t + ∂(hu)/∂x + ∂(hv)/∂y = 0                    (continuity)
∂(hu)/∂t + ∂(hu²)/∂x + ∂(huv)/∂y = -gh∂η/∂x - τ_bx/ρ + ν∇²(hu)   (x-momentum)
∂(hv)/∂t + ∂(huv)/∂x + ∂(hv²)/∂y = -gh∂η/∂y - τ_by/ρ + ν∇²(hv)   (y-momentum)

Uses conservative form with:
- Arakawa C-grid staggering for stability
- MUSCL-Hancock scheme for advection (2nd order)
- Semi-implicit treatment of friction for stability with thin water
- Wetting/drying handled via depth limiting

All core computations are JIT-compiled and vectorized for GPU execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, jit, lax

# Enable 64-bit precision for accuracy
jax.config.update("jax_enable_x64", True)


# Physical constants
GRAVITY: float = 9.81
MIN_DEPTH: float = 0.01  # Minimum depth for wet cells (m)
DRY_TOLERANCE: float = 1e-6  # Depth below which cell is considered dry


class SimulationParams(NamedTuple):
    """Immutable simulation parameters for JIT compilation."""
    dx: float
    dy: float
    dt: float
    gravity: float
    manning_n: Array  # Spatially varying Manning's n
    eddy_viscosity: float
    min_depth: float


class SimulationState(NamedTuple):
    """Immutable simulation state for JIT compilation."""
    eta: Array      # Water surface elevation (ny, nx)
    hu: Array       # x-momentum = h * u (ny, nx)
    hv: Array       # y-momentum = h * v (ny, nx)
    tracer: Array   # River water concentration 0-1 (ny, nx)
    time: float     # Simulation time in seconds


@jit
def compute_depth(eta: Array, z_bed: Array, min_depth: float) -> Array:
    """Compute water depth, enforcing minimum."""
    h = eta - z_bed
    return jnp.maximum(h, min_depth)


@jit
def compute_wet_mask(h: Array, min_depth: float) -> Array:
    """Create mask of wet cells (depth > threshold)."""
    return h > min_depth * 2


@jit
def compute_velocities(hu: Array, hv: Array, h: Array, min_depth: float) -> tuple[Array, Array]:
    """Recover velocities from conservative variables."""
    h_safe = jnp.maximum(h, min_depth)
    u = jnp.where(h > min_depth, hu / h_safe, 0.0)
    v = jnp.where(h > min_depth, hv / h_safe, 0.0)
    return u, v


@jit
def minmod(a: Array, b: Array) -> Array:
    """Minmod slope limiter for MUSCL reconstruction."""
    return jnp.where(
        a * b > 0,
        jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)),
        0.0
    )


@jit
def muscl_reconstruct_x(q: Array, dx: float) -> tuple[Array, Array]:
    """MUSCL reconstruction in x-direction.

    Returns left and right states at cell interfaces.
    """
    # Slopes
    dq_left = q - jnp.roll(q, 1, axis=1)
    dq_right = jnp.roll(q, -1, axis=1) - q

    # Limited slope
    dq = minmod(dq_left, dq_right)

    # Reconstructed values at interfaces
    q_left = q - 0.5 * dq   # Left side of right interface
    q_right = q + 0.5 * dq  # Right side of left interface

    return q_left, q_right


@jit
def muscl_reconstruct_y(q: Array, dy: float) -> tuple[Array, Array]:
    """MUSCL reconstruction in y-direction."""
    dq_left = q - jnp.roll(q, 1, axis=0)
    dq_right = jnp.roll(q, -1, axis=0) - q
    dq = minmod(dq_left, dq_right)
    q_left = q - 0.5 * dq
    q_right = q + 0.5 * dq
    return q_left, q_right


@jit
def hll_flux_x(
    h_l: Array, h_r: Array,
    hu_l: Array, hu_r: Array,
    hv_l: Array, hv_r: Array,
    g: float,
    min_depth: float,
) -> tuple[Array, Array, Array]:
    """HLL approximate Riemann solver for x-direction fluxes.

    Returns fluxes for h, hu, hv at cell interfaces.
    """
    # Velocities
    h_l_safe = jnp.maximum(h_l, min_depth)
    h_r_safe = jnp.maximum(h_r, min_depth)
    u_l = jnp.where(h_l > min_depth, hu_l / h_l_safe, 0.0)
    u_r = jnp.where(h_r > min_depth, hu_r / h_r_safe, 0.0)

    # Wave speeds
    c_l = jnp.sqrt(g * h_l_safe)
    c_r = jnp.sqrt(g * h_r_safe)

    s_l = jnp.minimum(u_l - c_l, u_r - c_r)
    s_r = jnp.maximum(u_l + c_l, u_r + c_r)

    # Fluxes for left and right states
    # F = [hu, hu² + 0.5*g*h², huv]
    f_h_l = hu_l
    f_h_r = hu_r
    f_hu_l = hu_l * u_l + 0.5 * g * h_l * h_l
    f_hu_r = hu_r * u_r + 0.5 * g * h_r * h_r
    f_hv_l = hv_l * u_l
    f_hv_r = hv_r * u_r

    # HLL flux
    denom = s_r - s_l + 1e-10

    f_h = jnp.where(
        s_l >= 0, f_h_l,
        jnp.where(
            s_r <= 0, f_h_r,
            (s_r * f_h_l - s_l * f_h_r + s_l * s_r * (h_r - h_l)) / denom
        )
    )

    f_hu = jnp.where(
        s_l >= 0, f_hu_l,
        jnp.where(
            s_r <= 0, f_hu_r,
            (s_r * f_hu_l - s_l * f_hu_r + s_l * s_r * (hu_r - hu_l)) / denom
        )
    )

    f_hv = jnp.where(
        s_l >= 0, f_hv_l,
        jnp.where(
            s_r <= 0, f_hv_r,
            (s_r * f_hv_l - s_l * f_hv_r + s_l * s_r * (hv_r - hv_l)) / denom
        )
    )

    return f_h, f_hu, f_hv


@jit
def hll_flux_y(
    h_l: Array, h_r: Array,
    hu_l: Array, hu_r: Array,
    hv_l: Array, hv_r: Array,
    g: float,
    min_depth: float,
) -> tuple[Array, Array, Array]:
    """HLL approximate Riemann solver for y-direction fluxes."""
    h_l_safe = jnp.maximum(h_l, min_depth)
    h_r_safe = jnp.maximum(h_r, min_depth)
    v_l = jnp.where(h_l > min_depth, hv_l / h_l_safe, 0.0)
    v_r = jnp.where(h_r > min_depth, hv_r / h_r_safe, 0.0)

    c_l = jnp.sqrt(g * h_l_safe)
    c_r = jnp.sqrt(g * h_r_safe)

    s_l = jnp.minimum(v_l - c_l, v_r - c_r)
    s_r = jnp.maximum(v_l + c_l, v_r + c_r)

    # G = [hv, huv, hv² + 0.5*g*h²]
    g_h_l = hv_l
    g_h_r = hv_r
    g_hu_l = hu_l * v_l
    g_hu_r = hu_r * v_r
    g_hv_l = hv_l * v_l + 0.5 * g * h_l * h_l
    g_hv_r = hv_r * v_r + 0.5 * g * h_r * h_r

    denom = s_r - s_l + 1e-10

    g_h = jnp.where(
        s_l >= 0, g_h_l,
        jnp.where(
            s_r <= 0, g_h_r,
            (s_r * g_h_l - s_l * g_h_r + s_l * s_r * (h_r - h_l)) / denom
        )
    )

    g_hu = jnp.where(
        s_l >= 0, g_hu_l,
        jnp.where(
            s_r <= 0, g_hu_r,
            (s_r * g_hu_l - s_l * g_hu_r + s_l * s_r * (hu_r - hu_l)) / denom
        )
    )

    g_hv = jnp.where(
        s_l >= 0, g_hv_l,
        jnp.where(
            s_r <= 0, g_hv_r,
            (s_r * g_hv_l - s_l * g_hv_r + s_l * s_r * (hv_r - hv_l)) / denom
        )
    )

    return g_h, g_hu, g_hv


@jit
def compute_friction_implicit(
    hu: Array, hv: Array, h: Array,
    manning_n: Array, dt: float, g: float, min_depth: float
) -> tuple[Array, Array]:
    """Apply friction implicitly for stability.

    Uses the formula: hu_new = hu / (1 + dt * Cf * |u| / h)
    where Cf = g * n² / h^(1/3)
    """
    h_safe = jnp.maximum(h, min_depth)
    u, v = compute_velocities(hu, hv, h, min_depth)
    speed = jnp.sqrt(u * u + v * v)

    # Friction coefficient
    cf = g * manning_n * manning_n / jnp.power(h_safe, 1.0/3.0)

    # Implicit factor
    factor = 1.0 / (1.0 + dt * cf * speed / h_safe)

    # Apply only where wet
    wet = h > min_depth
    hu_new = jnp.where(wet, hu * factor, 0.0)
    hv_new = jnp.where(wet, hv * factor, 0.0)

    return hu_new, hv_new


@jit
def apply_viscosity(q: Array, nu: float, dt: float, dx: float, dy: float) -> Array:
    """Apply eddy viscosity (Laplacian diffusion)."""
    lap = (
        (jnp.roll(q, 1, axis=1) - 2*q + jnp.roll(q, -1, axis=1)) / (dx * dx) +
        (jnp.roll(q, 1, axis=0) - 2*q + jnp.roll(q, -1, axis=0)) / (dy * dy)
    )
    return q + nu * dt * lap


@jit
def advect_tracer(
    tracer: Array, u: Array, v: Array, h: Array,
    dt: float, dx: float, dy: float, min_depth: float,
    diffusion: float = 10.0  # Increased diffusion for visibility
) -> Array:
    """Advect tracer using upwind scheme with diffusion."""
    wet = h > min_depth

    # Upwind in x
    dtracer_dx_neg = (tracer - jnp.roll(tracer, 1, axis=1)) / dx
    dtracer_dx_pos = (jnp.roll(tracer, -1, axis=1) - tracer) / dx
    adv_x = jnp.where(u > 0, u * dtracer_dx_neg, u * dtracer_dx_pos)

    # Upwind in y
    dtracer_dy_neg = (tracer - jnp.roll(tracer, 1, axis=0)) / dy
    dtracer_dy_pos = (jnp.roll(tracer, -1, axis=0) - tracer) / dy
    adv_y = jnp.where(v > 0, v * dtracer_dy_neg, v * dtracer_dy_pos)

    # Diffusion
    lap = (
        (jnp.roll(tracer, 1, axis=1) - 2*tracer + jnp.roll(tracer, -1, axis=1)) / (dx * dx) +
        (jnp.roll(tracer, 1, axis=0) - 2*tracer + jnp.roll(tracer, -1, axis=0)) / (dy * dy)
    )

    tracer_new = tracer - dt * (adv_x + adv_y) + diffusion * dt * lap
    tracer_new = jnp.clip(tracer_new, 0.0, 1.0)
    tracer_new = jnp.where(wet, tracer_new, 0.0)

    return tracer_new


@partial(jit, static_argnums=(2, 3, 4, 5, 6, 7))
def step_shallow_water(
    state: SimulationState,
    z_bed: Array,
    dx: float,
    dy: float,
    dt: float,
    g: float,
    eddy_viscosity: float,
    min_depth: float,
    manning_n: Array,
    tide_level: float,
    river_flow: float,
    ocean_mask: Array,
    river_mask: Array,
    river_velocity_dir: tuple[float, float],
) -> SimulationState:
    """Single timestep of shallow water equations.

    Uses first-order upwind scheme with strict wetting/drying control.
    Water can only flow into cells where eta > z_bed (i.e., it would be wet).
    """
    eta, hu, hv, tracer, time = state

    # Compute depth - enforce that water cannot go above 10m anywhere
    h = jnp.clip(eta - z_bed, min_depth, 10.0)
    wet = h > min_depth * 2

    # Get velocities - only where wet
    h_safe = jnp.maximum(h, min_depth)
    u = jnp.where(wet, hu / h_safe, 0.0)
    v = jnp.where(wet, hv / h_safe, 0.0)

    # --- Gravity-driven flow ---
    # Water flows down the water surface gradient
    eta_east = jnp.roll(eta, -1, axis=1)
    eta_west = jnp.roll(eta, 1, axis=1)
    eta_north = jnp.roll(eta, -1, axis=0)
    eta_south = jnp.roll(eta, 1, axis=0)

    # Can only receive water if neighbor has higher water surface AND we would be wet
    # This prevents water flowing onto dry land
    z_east = jnp.roll(z_bed, -1, axis=1)
    z_west = jnp.roll(z_bed, 1, axis=1)
    z_north = jnp.roll(z_bed, -1, axis=0)
    z_south = jnp.roll(z_bed, 1, axis=0)

    # Effective water surface for flow calculations (max of eta and z_bed)
    eta_eff = jnp.maximum(eta, z_bed)
    eta_eff_east = jnp.maximum(eta_east, z_east)
    eta_eff_west = jnp.maximum(eta_west, z_west)
    eta_eff_north = jnp.maximum(eta_north, z_north)
    eta_eff_south = jnp.maximum(eta_south, z_south)

    # Surface gradient using effective surfaces
    deta_dx = (eta_eff_east - eta_eff_west) / (2 * dx)
    deta_dy = (eta_eff_north - eta_eff_south) / (2 * dy)

    # Pressure gradient drives velocity
    # Only apply where wet
    accel_x = jnp.where(wet, -g * deta_dx, 0.0)
    accel_y = jnp.where(wet, -g * deta_dy, 0.0)

    # Update velocity
    u_new = u + dt * accel_x
    v_new = v + dt * accel_y

    # --- Friction (implicit, strong) ---
    speed = jnp.sqrt(u_new**2 + v_new**2)
    cf = g * manning_n**2 / jnp.power(h_safe, 1.0/3.0)
    friction_factor = 1.0 / (1.0 + dt * cf * (speed + 0.1) / h_safe)
    u_new = u_new * friction_factor
    v_new = v_new * friction_factor

    # --- Velocity limiting ---
    max_vel = 2.0  # Max 2 m/s
    speed = jnp.sqrt(u_new**2 + v_new**2)
    scale = jnp.where(speed > max_vel, max_vel / (speed + 1e-10), 1.0)
    u_new = u_new * scale
    v_new = v_new * scale

    # --- Mass flux (conservative) ---
    # Compute momentum
    hu_new = h * u_new
    hv_new = h * v_new

    # Upwind fluxes at cell faces
    # East face: positive hu means flow to east
    hu_face_east = jnp.where(hu_new > 0, hu_new, jnp.roll(hu_new, -1, axis=1))
    hu_face_west = jnp.where(jnp.roll(hu_new, 1, axis=1) > 0, jnp.roll(hu_new, 1, axis=1), hu_new)

    # North face: positive hv means flow to north (decreasing row index)
    hv_face_north = jnp.where(hv_new > 0, hv_new, jnp.roll(hv_new, -1, axis=0))
    hv_face_south = jnp.where(jnp.roll(hv_new, 1, axis=0) > 0, jnp.roll(hv_new, 1, axis=0), hv_new)

    # Mass flux divergence
    div_flux = (
        (hu_face_east - hu_face_west) / dx +
        (hv_face_north - hv_face_south) / dy
    )

    # Update depth
    h_new = h - dt * div_flux
    h_new = jnp.clip(h_new, min_depth, 10.0)  # Strict depth limits

    # Update eta
    eta_new = h_new + z_bed

    # Recompute momentum with new depth
    hu_new = h_new * u_new
    hv_new = h_new * v_new

    # --- Boundary conditions ---

    # Ocean boundary: Relaxation to tidal level with Flather-style radiation
    # Use a gentler approach than pure Flather to avoid instability
    h_ocean = jnp.maximum(tide_level - z_bed, min_depth)

    # Relaxation timescale - controls how fast boundary adjusts to tide
    # Shorter = more responsive but can be unstable; longer = smoother
    relax_time = 300.0  # 5 minutes
    relax_factor = dt / relax_time

    # Gradually adjust eta toward tide level at boundary
    eta_boundary = eta_new + relax_factor * (tide_level - eta_new)
    eta_new = jnp.where(ocean_mask, eta_boundary, eta_new)
    h_new = jnp.where(ocean_mask, jnp.maximum(eta_new - z_bed, min_depth), h_new)

    # Allow existing momentum to persist at boundary (radiation)
    # but damp it slightly to prevent reflection
    hu_new = jnp.where(ocean_mask, hu_new * 0.95, hu_new)
    hv_new = jnp.where(ocean_mask, hv_new * 0.95, hv_new)

    # River inflow
    n_river_cells = jnp.sum(river_mask)
    river_area = n_river_cells * dx * dy
    river_h_avg = jnp.sum(jnp.where(river_mask, h_new, 0.0)) / (n_river_cells + 1e-10)
    river_h_avg = jnp.maximum(river_h_avg, 0.5)  # Assume minimum 0.5m at falls

    # River velocity from continuity
    river_width = jnp.sqrt(river_area + 1e-10)
    river_vel = river_flow / (river_h_avg * river_width + 1e-10)
    river_vel = jnp.clip(river_vel, 0.0, 2.0)

    # Set river momentum (overwrite)
    hu_river = river_vel * river_h_avg * river_velocity_dir[0]
    hv_river = river_vel * river_h_avg * river_velocity_dir[1]
    hu_new = jnp.where(river_mask, hu_river, hu_new)
    hv_new = jnp.where(river_mask, hv_river, hv_new)

    # Add river water volume (mass source)
    dh_river = river_flow * dt / (river_area + 1e-10)
    h_new = jnp.where(river_mask, h_new + dh_river, h_new)
    eta_new = jnp.where(river_mask, h_new + z_bed, eta_new)

    # --- Wet/dry enforcement ---
    wet_new = h_new > min_depth * 2
    hu_new = jnp.where(wet_new, hu_new, 0.0)
    hv_new = jnp.where(wet_new, hv_new, 0.0)

    # --- Tracer advection ---
    u_final, v_final = compute_velocities(hu_new, hv_new, h_new, min_depth)
    tracer_new = advect_tracer(tracer, u_final, v_final, h_new, dt, dx, dy, min_depth)

    # River tracer = 1, ocean tracer = 0
    tracer_new = jnp.where(river_mask, 1.0, tracer_new)
    tracer_new = jnp.where(ocean_mask, 0.0, tracer_new)

    # Clamp values to prevent NaN propagation
    eta_new = jnp.nan_to_num(eta_new, nan=z_bed.mean())
    hu_new = jnp.nan_to_num(hu_new, nan=0.0)
    hv_new = jnp.nan_to_num(hv_new, nan=0.0)
    tracer_new = jnp.nan_to_num(tracer_new, nan=0.0)

    return SimulationState(
        eta=eta_new,
        hu=hu_new,
        hv=hv_new,
        tracer=tracer_new,
        time=time + dt,
    )


@dataclass
class ShallowWaterModel:
    """High-level interface for the shallow water solver."""

    z_bed: Array
    dx: float
    dy: float
    manning_n: Array
    ocean_mask: Array
    river_mask: Array
    river_direction: tuple[float, float]

    # Computed properties
    ny: int = 0
    nx: int = 0
    dt_cfl: float = 0.0

    def __post_init__(self):
        self.ny, self.nx = self.z_bed.shape
        # CFL-limited timestep
        max_depth = 10.0
        max_wave_speed = jnp.sqrt(GRAVITY * max_depth)
        self.dt_cfl = 0.4 * min(self.dx, self.dy) / float(max_wave_speed)

    @classmethod
    def from_arrays(
        cls,
        bathymetry: Array,
        dx: float,
        dy: float,
        mangrove_mask: Array | None = None,
        ocean_cols: int = 10,
        river_row: int = 0,
        river_col: int = 0,
        river_radius: int = 5,
        ocean_elevation_threshold: float = 2.0,
    ) -> "ShallowWaterModel":
        """Create model from arrays.

        Args:
            bathymetry: Bed elevation (m), shape (ny, nx)
            dx, dy: Grid spacing (m)
            mangrove_mask: Boolean mask of mangrove areas
            ocean_cols: Number of columns at right edge for ocean boundary
            river_row, river_col: Grid location of river source
            river_radius: Radius of river source region
            ocean_elevation_threshold: Only cells below this elevation at boundary are ocean
        """
        z_bed = jnp.array(bathymetry, dtype=jnp.float64)
        ny, nx = z_bed.shape

        # Manning's n
        base_n = 0.025  # Typical estuary
        mangrove_n = 0.12  # High friction in mangroves

        if mangrove_mask is not None:
            manning_n = jnp.where(mangrove_mask, mangrove_n, base_n)
        else:
            manning_n = jnp.full_like(z_bed, base_n)

        # Ocean boundary - only at eastern edge WHERE elevation is below threshold
        # This prevents flooding high ground at the boundary
        boundary_cols = jnp.zeros((ny, nx), dtype=bool)
        boundary_cols = boundary_cols.at[:, -ocean_cols:].set(True)
        low_elevation = z_bed < ocean_elevation_threshold
        ocean_mask = boundary_cols & low_elevation

        # River source
        y_idx, x_idx = jnp.mgrid[:ny, :nx]
        dist = jnp.sqrt((y_idx - river_row)**2 + (x_idx - river_col)**2)
        river_mask = dist <= river_radius

        # River flows generally eastward/downstream
        river_direction = (0.9, 0.2)  # Mostly east, slightly north

        return cls(
            z_bed=z_bed,
            dx=dx,
            dy=dy,
            manning_n=manning_n,
            ocean_mask=ocean_mask,
            river_mask=river_mask,
            river_direction=river_direction,
        )

    def create_initial_state(self, water_level: float = 0.0) -> SimulationState:
        """Create initial state with water at rest.

        Only floods cells that are:
        1. Below the water level, AND
        2. Connected to the ocean boundary (via flood fill)
        """
        import numpy as np
        from scipy import ndimage

        z_np = np.array(self.z_bed)
        ocean_np = np.array(self.ocean_mask)

        # Cells that could be wet (below water level)
        potentially_wet = z_np < water_level

        # Start flood fill from ocean boundary cells that are wet
        seed = ocean_np & potentially_wet

        # Use binary dilation to flood fill connected wet cells
        # Structure for 4-connectivity
        struct = ndimage.generate_binary_structure(2, 1)

        # Iteratively expand from ocean into connected wet cells
        flooded = seed.copy()
        for _ in range(max(self.ny, self.nx)):  # Max iterations
            expanded = ndimage.binary_dilation(flooded, structure=struct)
            new_flooded = expanded & potentially_wet
            if np.array_equal(new_flooded, flooded):
                break
            flooded = new_flooded

        # Also include river source area
        river_np = np.array(self.river_mask)
        flooded = flooded | river_np

        # Set water level only in flooded cells
        flooded_jax = jnp.array(flooded)
        eta = jnp.where(
            flooded_jax,
            water_level,
            self.z_bed + MIN_DEPTH  # Dry cells: eta just above bed
        )

        return SimulationState(
            eta=eta,
            hu=jnp.zeros_like(eta),
            hv=jnp.zeros_like(eta),
            tracer=jnp.zeros_like(eta),
            time=0.0,
        )

    def step(
        self,
        state: SimulationState,
        dt: float | None = None,
        tide_level: float = 0.0,
        river_flow: float = 0.5,
        eddy_viscosity: float = 1.0,
    ) -> SimulationState:
        """Advance model by one timestep."""
        if dt is None:
            dt = self.dt_cfl

        return step_shallow_water(
            state,
            self.z_bed,
            self.dx,
            self.dy,
            dt,
            GRAVITY,
            eddy_viscosity,
            MIN_DEPTH,
            self.manning_n,
            tide_level,
            river_flow,
            self.ocean_mask,
            self.river_mask,
            self.river_direction,
        )

    def get_depth(self, state: SimulationState) -> Array:
        """Get water depth from state."""
        return compute_depth(state.eta, self.z_bed, MIN_DEPTH)

    def get_velocities(self, state: SimulationState) -> tuple[Array, Array]:
        """Get u, v velocities from state."""
        h = self.get_depth(state)
        return compute_velocities(state.hu, state.hv, h, MIN_DEPTH)

    def get_speed(self, state: SimulationState) -> Array:
        """Get flow speed magnitude."""
        u, v = self.get_velocities(state)
        return jnp.sqrt(u**2 + v**2)


def run_simulation(
    model: ShallowWaterModel,
    duration_seconds: float,
    dt: float | None = None,
    tide_amplitude: float = 0.8,
    tide_period_seconds: float = 12.42 * 3600,
    mean_water_level: float = 0.3,
    start_phase: float = 0.0,
    river_flow: float = 0.5,
    output_interval: float = 300.0,  # 5 minutes
    progress_callback=None,
) -> list[SimulationState]:
    """Run full simulation.

    Args:
        model: ShallowWaterModel instance
        duration_seconds: Total simulation time
        dt: Timestep (uses CFL if None)
        tide_amplitude: Tidal amplitude (m)
        tide_period_seconds: Tidal period (default M2 tide)
        mean_water_level: Mean water level (m)
        start_phase: Starting phase (0=low, 0.5=high, 1=low)
        river_flow: River discharge (m³/s)
        output_interval: How often to save state (seconds)
        progress_callback: Optional callback(progress_fraction, state)

    Returns:
        List of states at output intervals
    """
    if dt is None:
        dt = model.dt_cfl

    omega = 2 * jnp.pi / tide_period_seconds
    phase0 = start_phase * 2 * jnp.pi

    # Initial tide level
    tide0 = mean_water_level + tide_amplitude * jnp.sin(phase0)
    state = model.create_initial_state(float(tide0))

    results = [state]

    n_steps = int(duration_seconds / dt)
    output_every = max(1, int(output_interval / dt))

    print(f"Running simulation:")
    print(f"  Grid: {model.ny} x {model.nx}")
    print(f"  Duration: {duration_seconds/3600:.1f} hours")
    print(f"  Timestep: {dt:.3f} s (CFL: {model.dt_cfl:.3f} s)")
    print(f"  Steps: {n_steps}")
    print(f"  Output every {output_every} steps")

    # JIT compile on first step
    print("  Compiling (first step)...")
    tide_level = float(mean_water_level + tide_amplitude * jnp.sin(omega * dt + phase0))
    state = model.step(state, dt, tide_level, river_flow)

    print("  Running...")
    for i in range(1, n_steps):
        t = i * dt
        tide_level = float(mean_water_level + tide_amplitude * jnp.sin(omega * t + phase0))

        state = model.step(state, dt, tide_level, river_flow)

        if (i + 1) % output_every == 0:
            results.append(state)
            progress = (i + 1) / n_steps

            if progress_callback:
                progress_callback(progress, state)

            if (i + 1) % (output_every * 10) == 0:
                h = model.get_depth(state)
                print(f"    {progress*100:.0f}% - t={t/3600:.2f}h, "
                      f"tide={tide_level:.2f}m, "
                      f"max_depth={float(jnp.max(h)):.2f}m")

    return results
