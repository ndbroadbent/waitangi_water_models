# Tracer Advection in Shallow Water Simulations

## The Problem: Decoupled Fields Don't Work

A common approach in shallow water simulations is to track water and tracers (dye, salinity, pollutants) as separate 2D fields:

```python
h = ...            # Water depth at each cell
river_tracer = ... # River water concentration (0-1)
ocean_tracer = ... # Ocean water concentration (0-1)
```

Then advect them separately:
1. Update `h` based on water fluxes through cell faces
2. Separately advect `river_tracer` using the same fluxes
3. Separately advect `ocean_tracer` using the same fluxes

**This breaks at wetting fronts.**

When ocean water flows into a completely dry basin, some wet cells end up with tracer < 1.0 - appearing as "untraced" water. This is physically impossible: if the only water source is ocean, ALL wet cells must have ocean tracer = 1.0.

### Why It Fails

The tracer advection equation:
```
tracer_new = (tracer * h - dt * tracer_divergence) / h_new
```

When `h = 0` (dry cell becoming wet):
- `tracer * h = 0` (no old tracer mass)
- Must infer tracer purely from inflows
- But flux-based calculations at wetting fronts have numerical edge cases
- Cells becoming wet simultaneously can't "see" each other's tracer

The fundamental flaw: **water and tracer are treated as independent quantities that we try to synchronize after the fact.**

## The Solution: Unified Water-Tracer Field

Instead of separate fields, combine them:

```python
# OLD: Three separate fields
h = ...
river_tracer = ...
ocean_tracer = ...

# NEW: Unified field - tracer is an intrinsic property of water
water[:,:,0] = h                    # depth
water[:,:,1] = river_fraction       # intrinsic property
water[:,:,2] = ocean_fraction       # intrinsic property
```

When water moves, its tracer properties move with it **by definition** - they're the same thing.

### Why This Works

For a dry cell becoming wet:
- Water packets flow in from neighbors
- Each packet brings its tracer concentration
- Receiving cell's tracer = weighted average of incoming packet tracers
- **By construction**, if all incoming water is ocean (tracer=1.0), receiving cell gets tracer=1.0

## Implementation: Advect Mass, Not Concentration

The cleanest implementation tracks **tracer mass** (`tracer * h`) rather than concentration:

```python
@jit
def simulation_step(eta, u, v, river_mass, ocean_mass, z_bed, ...):
    h = jnp.maximum(eta - z_bed, 0.0)

    # Upwind water depth at cell faces
    h_e = jnp.where(u[:-1, :] > 0, h[:-1, :], h[1:, :])
    h_n = jnp.where(v[:, :-1] > 0, h[:, :-1], h[:, 1:])

    # Tracer CONCENTRATION at faces (derived from mass)
    h_safe = jnp.maximum(h, 1e-10)
    river_conc = river_mass / h_safe
    ocean_conc = ocean_mass / h_safe

    # Upwind concentrations
    river_conc_e = jnp.where(u[:-1, :] > 0, river_conc[:-1, :], river_conc[1:, :])
    ocean_conc_e = jnp.where(u[:-1, :] > 0, ocean_conc[:-1, :], ocean_conc[1:, :])
    # ... same for _n directions

    # Water flux through faces
    flux_e = u[:-1, :] * h_e
    flux_n = v[:, :-1] * h_n

    # Tracer MASS flux = water flux * concentration
    # This is the key: mass flux is coupled to water flux
    river_flux_e = flux_e * river_conc_e
    ocean_flux_e = flux_e * ocean_conc_e
    # ... same for _n

    # Update water (standard shallow water)
    div_water = compute_divergence(flux_e, flux_n, dx, dy)
    eta = eta - dt * div_water

    # Update tracer MASS using same divergence pattern
    div_river = compute_divergence(river_flux_e, river_flux_n, dx, dy)
    div_ocean = compute_divergence(ocean_flux_e, ocean_flux_n, dx, dy)

    river_mass = river_mass - dt * div_river
    ocean_mass = ocean_mass - dt * div_ocean

    # Source injection (add MASS, not concentration)
    h_new = jnp.maximum(eta - z_bed, 0.0)
    river_mass = jnp.where(river_source & (h_new > 0.01), h_new, river_mass)
    ocean_mass = jnp.where(ocean_source & (h_new > 0.01), h_new, ocean_mass)

    return eta, u, v, river_mass, ocean_mass
```

### Converting Back to Concentration

For rendering/output, convert mass back to concentration:

```python
h = jnp.maximum(eta - z_bed, 0.0)
h_safe = jnp.maximum(h, 1e-10)
river_tracer = jnp.where(h > 0.01, river_mass / h_safe, 0.0)
ocean_tracer = jnp.where(h > 0.01, ocean_mass / h_safe, 0.0)
```

## Key Insight

The tracer mass flux formula:
```
tracer_flux = water_flux * tracer_concentration
            = (u * h) * (tracer_mass / h)
            = u * tracer_mass
```

When we compute the divergence of tracer mass flux using the **same velocity field** as water, the tracer automatically stays coupled to the water. No special handling needed for dry-to-wet transitions.

## Benefits

1. **Physically correct by construction** - tracer mass conservation is guaranteed
2. **Simpler code** - no special cases for wetting fronts
3. **Extensible** - easy to add more tracers (salinity, temperature, pollutants)
4. **Better numerics** - no edge cases at wetting fronts

## Summary

| Approach | Track | Problem |
|----------|-------|---------|
| Separate fields | `h`, `tracer` independently | Decoupling at wetting fronts |
| Unified field | `h`, `tracer_mass = h * tracer` | None - mass moves with water |

The key mental model shift: **tracer is not a separate thing that follows water - it's an intrinsic property of the water itself.**
