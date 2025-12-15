"""Test velocity direction during tidal filling.

When ocean water floods a dry basin at high tide, water should flow from
east (ocean) to west (inland). This test verifies that velocity vectors
correctly show westward (negative u) flow during tidal filling.

Grid convention:
- u: eastward velocity (positive = east, negative = west)
- v: northward velocity (positive = north, negative = south)
- Column index increases eastward
- Row index increases southward (row 0 = north)
"""

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp

from waitangi.data.elevation import fetch_waitangi_elevation


@jax.jit
def simulation_step(eta, u, v, z_bed, manning_field, params, wall_mask, tide_level):
    """Simplified simulation step for testing (no tracers)."""
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
    Cf_u = g * manning_at_u**2 / jnp.power(h_at_u, 1/3)
    Cf_v = g * manning_at_v**2 / jnp.power(h_at_v, 1/3)
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
    eta = jnp.where(wall_mask, tide_level, eta)

    return eta, u, v


class TestTidalVelocityDirection:
    """Test velocity vectors during tidal filling of dry basin."""

    @pytest.fixture
    def simulation_setup(self):
        """Set up simulation with dry basin and high tide at east wall."""
        elev = fetch_waitangi_elevation()

        downsample = 8
        z_bed_np = elev.data[::downsample, ::downsample].copy()
        ny, nx = z_bed_np.shape
        dx = dy = abs(elev.transform.a) * downsample

        high_tide = 1.1

        # Wall at east boundary
        wall_col = 382 // downsample
        wall_mask_np = np.zeros((ny, nx), dtype=bool)
        for row in range(ny // 4, 3 * ny // 4):
            if z_bed_np[row, wall_col] < high_tide:
                wall_mask_np[row, wall_col] = True

        g = 9.81
        max_vel = 5.0
        H_max = high_tide + 3.0
        dt = 0.2 * min(dx, dy) / np.sqrt(g * H_max)
        manning_field = np.full((ny, nx), 0.035)

        z_bed = jnp.array(z_bed_np)
        wall_mask = jnp.array(wall_mask_np)
        manning_jax = jnp.array(manning_field)
        params = (dx, dy, dt, g, max_vel)

        # DRY START: basin empty, only wall has water
        eta_np = z_bed_np.copy()
        eta_np[wall_mask_np] = high_tide
        eta = jnp.array(eta_np)
        u = jnp.zeros((ny, nx))
        v = jnp.zeros((ny, nx))

        return {
            "eta": eta,
            "u": u,
            "v": v,
            "z_bed": z_bed,
            "z_bed_np": z_bed_np,
            "manning_field": manning_jax,
            "params": params,
            "wall_mask": wall_mask,
            "tide_level": high_tide,
            "dt": dt,
        }

    def test_velocity_after_5_minutes(self, simulation_setup):
        """After 5 minutes of tidal filling, verify specific velocity values.

        Expected values (from baseline run):
        - Wet cells: ~2450
        - Mean u (eastward): -0.42 m/s (NEGATIVE = westward flow)
        - Mean v (northward): +0.49 m/s
        - Mean speed: ~1.17 m/s
        - Westward flow fraction: ~66%
        """
        s = simulation_setup

        # Run for 5 minutes
        n_steps = int(5 * 60 / s["dt"])
        eta, u, v = s["eta"], s["u"], s["v"]

        for _ in range(n_steps):
            eta, u, v = simulation_step(
                eta, u, v, s["z_bed"], s["manning_field"],
                s["params"], s["wall_mask"], s["tide_level"]
            )

        u_np = np.array(u)
        v_np = np.array(v)
        eta_np = np.array(eta)
        h_np = np.maximum(eta_np - s["z_bed_np"], 0)

        wet_mask = h_np > 0.05
        speed = np.sqrt(u_np**2 + v_np**2)
        flowing_mask = wet_mask & (speed > 0.01)

        u_flowing = u_np[flowing_mask]
        v_flowing = v_np[flowing_mask]

        wet_cells = np.sum(wet_mask)
        flowing_cells = np.sum(flowing_mask)
        mean_u = np.mean(u_flowing)
        mean_v = np.mean(v_flowing)
        mean_speed = np.mean(speed[flowing_mask])
        westward_fraction = np.sum(u_flowing < 0) / len(u_flowing)

        # Verify expected values with tolerances
        assert 2000 < wet_cells < 3000, f"Expected ~2450 wet cells, got {wet_cells}"
        assert 2000 < flowing_cells < 3000, f"Expected ~2449 flowing cells, got {flowing_cells}"

        # Mean u should be negative (westward flow from ocean)
        assert -0.6 < mean_u < -0.2, f"Expected mean u ~ -0.42 m/s, got {mean_u:.4f}"

        # Mean v should be positive (northward component)
        assert 0.3 < mean_v < 0.7, f"Expected mean v ~ +0.49 m/s, got {mean_v:.4f}"

        # Mean speed should be around 1.17 m/s
        assert 0.8 < mean_speed < 1.6, f"Expected mean speed ~ 1.17 m/s, got {mean_speed:.4f}"

        # Majority should flow westward
        assert 0.60 < westward_fraction < 0.75, f"Expected ~66% westward, got {westward_fraction*100:.1f}%"

        print(f"\nVelocity test PASSED:")
        print(f"  Wet cells: {wet_cells}")
        print(f"  Mean u: {mean_u:.4f} m/s (westward)")
        print(f"  Mean v: {mean_v:.4f} m/s (northward)")
        print(f"  Mean speed: {mean_speed:.4f} m/s")
        print(f"  Westward fraction: {westward_fraction*100:.1f}%")
