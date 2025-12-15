"""HLL Riemann solver for 2D shallow water equations.

The HLL (Harten-Lax-van Leer) approximate Riemann solver computes numerical fluxes
at cell interfaces by considering two waves bounding an intermediate state.

For the shallow water equations:
  ∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = 0        (mass)
  ∂(hu)/∂t + ∂(hu² + gh²/2)/∂x + ∂(huv)/∂y = 0   (x-momentum)
  ∂(hv)/∂t + ∂(huv)/∂x + ∂(hv² + gh²/2)/∂y = 0   (y-momentum)

The HLL flux formula:
  F_HLL = (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / (S_R - S_L)

Where S_L and S_R are estimates of the fastest left/right-moving waves.

References:
- Toro, E.F. (2009). Riemann Solvers and Numerical Methods for Fluid Dynamics.
- LeVeque, R.J. (2002). Finite Volume Methods for Hyperbolic Problems.
"""

import jax.numpy as jnp
from jax import jit


@jit
def hll_shallow_water_flux_2d(
    h: jnp.ndarray,
    hu: jnp.ndarray,
    hv: jnp.ndarray,
    g: float,
    dx: float,
    dy: float,
) -> tuple[
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
]:
    """Compute HLL fluxes at all cell interfaces for 2D shallow water.

    Args:
        h: Water depth [ny, nx]
        hu: x-momentum (h * u) [ny, nx]
        hv: y-momentum (h * v) [ny, nx]
        g: Gravitational acceleration
        dx: Grid spacing in x
        dy: Grid spacing in y

    Returns:
        (flux_h_x, flux_hu_x, flux_hv_x): Mass and momentum fluxes at x-faces [ny, nx-1]
        (flux_h_y, flux_hu_y, flux_hv_y): Mass and momentum fluxes at y-faces [ny-1, nx]
    """
    # Ensure non-negative depth (numerical safety)
    h = jnp.maximum(h, 0.0)

    # Compute velocities (with protection for dry cells)
    h_safe = jnp.maximum(h, 1e-10)
    u = hu / h_safe
    v = hv / h_safe
    # Zero velocity in dry cells
    u = jnp.where(h > 1e-6, u, 0.0)
    v = jnp.where(h > 1e-6, v, 0.0)

    # Wave speed (celerity)
    c = jnp.sqrt(g * h_safe)
    c = jnp.where(h > 1e-6, c, 0.0)

    # ==== X-direction fluxes (at east faces) ====
    # Left state (cell i) and right state (cell i+1)
    h_L = h[:-1, :]
    h_R = h[1:, :]
    hu_L = hu[:-1, :]
    hu_R = hu[1:, :]
    hv_L = hv[:-1, :]
    hv_R = hv[1:, :]
    u_L = u[:-1, :]
    u_R = u[1:, :]
    c_L = c[:-1, :]
    c_R = c[1:, :]

    # Wave speed estimates (Einfeldt)
    s_L_x = jnp.minimum(u_L - c_L, u_R - c_R)
    s_R_x = jnp.maximum(u_L + c_L, u_R + c_R)

    # Ensure proper wave ordering for dry states
    s_L_x = jnp.minimum(s_L_x, 0.0)
    s_R_x = jnp.maximum(s_R_x, 0.0)

    # Physical fluxes in x-direction: F = [hu, hu² + gh²/2, huv]
    flux_h_L_x = hu_L
    flux_hu_L_x = hu_L * u_L + 0.5 * g * h_L**2
    flux_hv_L_x = hu_L * (hv_L / jnp.maximum(h_L, 1e-10))

    flux_h_R_x = hu_R
    flux_hu_R_x = hu_R * u_R + 0.5 * g * h_R**2
    flux_hv_R_x = hu_R * (hv_R / jnp.maximum(h_R, 1e-10))

    # HLL flux formula
    denom_x = s_R_x - s_L_x
    denom_x = jnp.where(jnp.abs(denom_x) < 1e-10, 1e-10, denom_x)

    flux_h_x = (
        s_R_x * flux_h_L_x - s_L_x * flux_h_R_x + s_L_x * s_R_x * (h_R - h_L)
    ) / denom_x
    flux_hu_x = (
        s_R_x * flux_hu_L_x - s_L_x * flux_hu_R_x + s_L_x * s_R_x * (hu_R - hu_L)
    ) / denom_x
    flux_hv_x = (
        s_R_x * flux_hv_L_x - s_L_x * flux_hv_R_x + s_L_x * s_R_x * (hv_R - hv_L)
    ) / denom_x

    # Degenerate case: both waves same sign -> use upwind
    flux_h_x = jnp.where(s_L_x >= 0, flux_h_L_x, flux_h_x)
    flux_h_x = jnp.where(s_R_x <= 0, flux_h_R_x, flux_h_x)
    flux_hu_x = jnp.where(s_L_x >= 0, flux_hu_L_x, flux_hu_x)
    flux_hu_x = jnp.where(s_R_x <= 0, flux_hu_R_x, flux_hu_x)
    flux_hv_x = jnp.where(s_L_x >= 0, flux_hv_L_x, flux_hv_x)
    flux_hv_x = jnp.where(s_R_x <= 0, flux_hv_R_x, flux_hv_x)

    # ==== Y-direction fluxes (at north faces) ====
    # Left state (cell j) and right state (cell j+1)
    h_L = h[:, :-1]
    h_R = h[:, 1:]
    hu_L = hu[:, :-1]
    hu_R = hu[:, 1:]
    hv_L = hv[:, :-1]
    hv_R = hv[:, 1:]
    v_L = v[:, :-1]
    v_R = v[:, 1:]
    c_L = c[:, :-1]
    c_R = c[:, 1:]

    # Wave speed estimates
    s_L_y = jnp.minimum(v_L - c_L, v_R - c_R)
    s_R_y = jnp.maximum(v_L + c_L, v_R + c_R)

    s_L_y = jnp.minimum(s_L_y, 0.0)
    s_R_y = jnp.maximum(s_R_y, 0.0)

    # Physical fluxes in y-direction: G = [hv, huv, hv² + gh²/2]
    flux_h_L_y = hv_L
    flux_hu_L_y = hv_L * (hu_L / jnp.maximum(h_L, 1e-10))
    flux_hv_L_y = hv_L * v_L + 0.5 * g * h_L**2

    flux_h_R_y = hv_R
    flux_hu_R_y = hv_R * (hu_R / jnp.maximum(h_R, 1e-10))
    flux_hv_R_y = hv_R * v_R + 0.5 * g * h_R**2

    # HLL flux formula
    denom_y = s_R_y - s_L_y
    denom_y = jnp.where(jnp.abs(denom_y) < 1e-10, 1e-10, denom_y)

    flux_h_y = (
        s_R_y * flux_h_L_y - s_L_y * flux_h_R_y + s_L_y * s_R_y * (h_R - h_L)
    ) / denom_y
    flux_hu_y = (
        s_R_y * flux_hu_L_y - s_L_y * flux_hu_R_y + s_L_y * s_R_y * (hu_R - hu_L)
    ) / denom_y
    flux_hv_y = (
        s_R_y * flux_hv_L_y - s_L_y * flux_hv_R_y + s_L_y * s_R_y * (hv_R - hv_L)
    ) / denom_y

    # Degenerate case: both waves same sign -> use upwind
    flux_h_y = jnp.where(s_L_y >= 0, flux_h_L_y, flux_h_y)
    flux_h_y = jnp.where(s_R_y <= 0, flux_h_R_y, flux_h_y)
    flux_hu_y = jnp.where(s_L_y >= 0, flux_hu_L_y, flux_hu_y)
    flux_hu_y = jnp.where(s_R_y <= 0, flux_hu_R_y, flux_hu_y)
    flux_hv_y = jnp.where(s_L_y >= 0, flux_hv_L_y, flux_hv_y)
    flux_hv_y = jnp.where(s_R_y <= 0, flux_hv_R_y, flux_hv_y)

    return (flux_h_x, flux_hu_x, flux_hv_x), (flux_h_y, flux_hu_y, flux_hv_y)


@jit
def update_conserved(
    h: jnp.ndarray,
    hu: jnp.ndarray,
    hv: jnp.ndarray,
    flux_x: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    flux_y: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    z_bed: jnp.ndarray,
    dt: float,
    dx: float,
    dy: float,
    g: float,
    manning: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Update conserved variables using HLL fluxes.

    Args:
        h, hu, hv: Current conserved state
        flux_x: (flux_h, flux_hu, flux_hv) at x-faces
        flux_y: (flux_h, flux_hu, flux_hv) at y-faces
        z_bed: Bed elevation
        dt: Time step
        dx, dy: Grid spacing
        g: Gravity
        manning: Manning roughness coefficient

    Returns:
        Updated (h, hu, hv)
    """
    flux_h_x, flux_hu_x, flux_hv_x = flux_x
    flux_h_y, flux_hu_y, flux_hv_y = flux_y

    # Compute divergence of fluxes
    # dF/dx at cell centers
    div_h = jnp.zeros_like(h)
    div_hu = jnp.zeros_like(h)
    div_hv = jnp.zeros_like(h)

    # X-fluxes: flux_*_x has shape [ny, nx-1]
    div_h = div_h.at[:-1, :].add(flux_h_x / dx)
    div_h = div_h.at[1:, :].add(-flux_h_x / dx)

    div_hu = div_hu.at[:-1, :].add(flux_hu_x / dx)
    div_hu = div_hu.at[1:, :].add(-flux_hu_x / dx)

    div_hv = div_hv.at[:-1, :].add(flux_hv_x / dx)
    div_hv = div_hv.at[1:, :].add(-flux_hv_x / dx)

    # Y-fluxes: flux_*_y has shape [ny-1, nx]
    div_h = div_h.at[:, :-1].add(flux_h_y / dy)
    div_h = div_h.at[:, 1:].add(-flux_h_y / dy)

    div_hu = div_hu.at[:, :-1].add(flux_hu_y / dy)
    div_hu = div_hu.at[:, 1:].add(-flux_hu_y / dy)

    div_hv = div_hv.at[:, :-1].add(flux_hv_y / dy)
    div_hv = div_hv.at[:, 1:].add(-flux_hv_y / dy)

    # Update conserved variables
    h_new = h - dt * div_h
    hu_new = hu - dt * div_hu
    hv_new = hv - dt * div_hv

    # Enforce positivity of depth
    h_new = jnp.maximum(h_new, 0.0)

    # Source terms: bed slope
    # S_b = -gh * ∂z/∂x, -gh * ∂z/∂y
    dz_dx = jnp.zeros_like(z_bed)
    dz_dy = jnp.zeros_like(z_bed)

    dz_dx = dz_dx.at[1:-1, :].set((z_bed[2:, :] - z_bed[:-2, :]) / (2 * dx))
    dz_dy = dz_dy.at[:, 1:-1].set((z_bed[:, 2:] - z_bed[:, :-2]) / (2 * dy))

    hu_new = hu_new - dt * g * h_new * dz_dx
    hv_new = hv_new - dt * g * h_new * dz_dy

    # Friction source term (semi-implicit for stability)
    h_safe = jnp.maximum(h_new, 1e-6)
    u_new = hu_new / h_safe
    v_new = hv_new / h_safe
    speed = jnp.sqrt(u_new**2 + v_new**2)

    # Manning friction coefficient
    Cf = g * manning**2 / jnp.power(h_safe, 1.0 / 3.0)

    # Semi-implicit friction: (1 + dt * Cf * |u| / h)
    friction_denom = 1.0 + dt * Cf * speed / h_safe
    hu_new = hu_new / friction_denom
    hv_new = hv_new / friction_denom

    # Zero momentum in dry cells
    hu_new = jnp.where(h_new > 1e-6, hu_new, 0.0)
    hv_new = jnp.where(h_new > 1e-6, hv_new, 0.0)

    return h_new, hu_new, hv_new
