#!/usr/bin/env python3
"""Simple tidal fill test - closed basin with wall at bay entrance.

Creates an artificial wall east of the bridge to make a closed basin,
then fills it with water at a fixed level from the wall.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from waitangi.data.elevation import fetch_waitangi_elevation, compute_flooded_area_at_level


def run_simple_fill_test():
    """Test filling a closed basin."""

    print("=" * 60)
    print("Closed Basin Fill Test")
    print("=" * 60)

    # Load elevation data
    print("\nLoading elevation data...")
    elev = fetch_waitangi_elevation()

    # Downsample for speed
    downsample = 8
    z_bed = elev.data[::downsample, ::downsample].copy()
    ny, nx = z_bed.shape
    dx = dy = abs(elev.transform.a) * downsample

    print(f"Grid: {ny} x {nx} cells, {dx:.0f}m resolution")

    # Create a wall to close off the bay
    # The bridge is roughly at column 320 (at 8m downsample from col 2558)
    # Put wall at column 350 to include some of the bay
    wall_col = 350
    wall_height = 20.0  # meters - high enough to block all water

    print(f"\nAdding wall at column {wall_col} (elevation {wall_height}m)")
    z_bed[:, wall_col:] = wall_height  # Everything east of wall is high ground

    # Also close north and south boundaries
    z_bed[:10, :] = wall_height
    z_bed[-10:, :] = wall_height

    # Test at high tide level
    tide_level = 1.0  # meters

    # Source cells - where water enters at fixed level (at the wall)
    # This is like a weir or spillway
    source_col = wall_col - 5
    source_mask = np.zeros((ny, nx), dtype=bool)
    # Only source where elevation is below tide level
    source_mask[:, source_col:source_col+3] = z_bed[:, source_col:source_col+3] < tide_level
    n_source = np.sum(source_mask)
    print(f"Source cells (at wall): {n_source}")

    # What SHOULD be flooded in this closed basin?
    # Everything below tide_level that's connected to the source
    from scipy import ndimage
    potentially_wet = z_bed < tide_level
    struct = ndimage.generate_binary_structure(2, 1)
    flooded = source_mask.copy()
    for _ in range(max(ny, nx)):
        expanded = ndimage.binary_dilation(flooded, structure=struct)
        new_flooded = expanded & potentially_wet
        if np.array_equal(new_flooded, flooded):
            break
        flooded = new_flooded
    expected_flood = flooded
    expected_area_km2 = np.sum(expected_flood) * dx * dy / 1e6
    print(f"Expected flooded area at {tide_level}m: {expected_area_km2:.2f} km²")

    # Physical parameters
    g = 9.81

    # Initialize: start empty, let it fill from source
    eta = z_bed.copy()  # Start dry
    eta[source_mask] = tide_level  # Source at tide level
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))

    # Compute initial depth
    h = np.maximum(eta - z_bed, 0.0)
    initial_wet = h > 0.01
    initial_area = np.sum(initial_wet) * dx * dy / 1e6
    print(f"Initial wet area: {initial_area:.2f} km²")

    # CFL timestep - use expected max depth
    H_max = tide_level + 2.0  # Conservative estimate
    dt = 0.2 * min(dx, dy) / np.sqrt(g * H_max)
    print(f"Timestep: {dt:.2f}s (CFL limited)")

    # Friction coefficient (Manning)
    manning_n = 0.035

    # Run simulation - enough steps to reach equilibrium
    n_steps = 20000
    print(f"\nRunning {n_steps} steps (simulating {n_steps*dt/60:.1f} minutes)...")

    for step in range(n_steps):
        # Compute depth
        h = np.maximum(eta - z_bed, 0.0)
        wet = h > 0.01

        # Update velocities based on surface gradient (only where wet)
        deta_dx = np.zeros_like(eta)
        deta_dx[:-1, :] = (eta[1:, :] - eta[:-1, :]) / dx

        deta_dy = np.zeros_like(eta)
        deta_dy[:, :-1] = (eta[:, 1:] - eta[:, :-1]) / dy

        # Acceleration: du/dt = -g * d(eta)/dx
        u[:-1, :] -= g * dt * deta_dx[:-1, :]
        v[:, :-1] -= g * dt * deta_dy[:, :-1]

        # Friction (implicit) - only where there's water
        h_at_u = np.maximum((h[:-1, :] + h[1:, :]) / 2, 0.01)  # depth at u-faces
        h_at_v = np.maximum((h[:, :-1] + h[:, 1:]) / 2, 0.01)  # depth at v-faces

        speed_u = np.abs(u[:-1, :])
        speed_v = np.abs(v[:, :-1])

        Cf_u = g * manning_n**2 / np.power(h_at_u, 1/3)
        Cf_v = g * manning_n**2 / np.power(h_at_v, 1/3)

        friction_u = 1.0 / (1.0 + dt * Cf_u * speed_u / h_at_u)
        friction_v = 1.0 / (1.0 + dt * Cf_v * speed_v / h_at_v)

        u[:-1, :] *= friction_u
        v[:, :-1] *= friction_v

        # Velocity limiting
        max_vel = 3.0
        u = np.clip(u, -max_vel, max_vel)
        v = np.clip(v, -max_vel, max_vel)

        # Zero velocity at boundaries (walls)
        u[-1, :] = 0.0
        u[0, :] = 0.0
        v[:, -1] = 0.0
        v[:, 0] = 0.0

        # Upwind scheme for mass flux
        h_e = np.zeros_like(h)
        h_e[:-1, :] = np.where(u[:-1, :] > 0, h[:-1, :], h[1:, :])

        h_n = np.zeros_like(h)
        h_n[:, :-1] = np.where(v[:, :-1] > 0, h[:, :-1], h[:, 1:])

        # Flux divergence (simplified)
        flux_e = u[:-1, :] * h_e[:-1, :]  # flux at east face of cell [:-1,:]
        flux_n = v[:, :-1] * h_n[:, :-1]  # flux at north face of cell [:,:-1]

        # Divergence: (flux_out_east - flux_in_west)/dx + (flux_out_north - flux_in_south)/dy
        div = np.zeros_like(eta)
        div[:-1, :] += flux_e / dx  # outflow to east
        div[1:, :] -= flux_e / dx   # inflow from west
        div[:, :-1] += flux_n / dy  # outflow to north
        div[:, 1:] -= flux_n / dy   # inflow from south

        # Update eta
        eta = eta - dt * div

        # Enforce: eta cannot go below z_bed
        eta = np.maximum(eta, z_bed)

        # ENFORCE SOURCE: hold water level fixed at source
        eta[source_mask] = tide_level

        # Progress
        if (step + 1) % 500 == 0:
            h = np.maximum(eta - z_bed, 0.0)
            wet = h > 0.01
            area = np.sum(wet) * dx * dy / 1e6
            max_speed = np.sqrt(np.max(u**2 + v**2))
            max_h = np.max(h)
            print(f"  Step {step+1}: area={area:.2f}km², max_h={max_h:.2f}m, max_speed={max_speed:.2f}m/s")

    # Final analysis
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    h_final = np.maximum(eta - z_bed, 0.0)
    wet_final = h_final > 0.01
    final_area = np.sum(wet_final) * dx * dy / 1e6

    print(f"Expected area: {expected_area_km2:.2f} km²")
    print(f"Final area:    {final_area:.2f} km²")
    print(f"Difference:    {final_area - expected_area_km2:.3f} km²")

    # Check for overflow
    overflow = wet_final & ~expected_flood
    overflow_area = np.sum(overflow) * dx * dy / 1e6
    print(f"\nOverflow (water outside expected): {overflow_area:.4f} km²")

    # Check for underfill
    underfill = expected_flood & ~wet_final
    underfill_area = np.sum(underfill) * dx * dy / 1e6
    print(f"Underfill (expected but dry):      {underfill_area:.4f} km²")

    max_depth = np.max(h_final)
    print(f"\nMax water depth: {max_depth:.2f}m")

    # Save comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    extent = [
        elev.bounds[0], elev.bounds[2],
        elev.bounds[1], elev.bounds[3]
    ]

    # Expected flood
    axes[0].imshow(expected_flood, origin='upper', extent=extent, cmap='Blues')
    axes[0].set_title(f'Expected Flood at {tide_level}m')
    axes[0].set_xlabel('Easting (m)')
    axes[0].set_ylabel('Northing (m)')

    # Simulated flood
    axes[1].imshow(wet_final, origin='upper', extent=extent, cmap='Blues')
    axes[1].set_title(f'Simulated Flood (after {n_steps} steps)')
    axes[1].set_xlabel('Easting (m)')

    # Difference
    diff = np.zeros_like(wet_final, dtype=int)
    diff[overflow] = 1  # Red: overflow
    diff[underfill] = -1  # Yellow: underfill
    im = axes[2].imshow(diff, origin='upper', extent=extent, cmap='RdYlBu', vmin=-1, vmax=1)
    axes[2].set_title('Difference (red=overflow, blue=underfill)')
    axes[2].set_xlabel('Easting (m)')

    plt.tight_layout()
    plt.savefig('simple_tidal_test.png', dpi=150)
    print(f"\nSaved plot to simple_tidal_test.png")

    return overflow_area < 0.01  # Success if minimal overflow


if __name__ == "__main__":
    success = run_simple_fill_test()
    print(f"\n{'SUCCESS' if success else 'FAILED'}: Water {'stayed' if success else 'did not stay'} within expected bounds")
