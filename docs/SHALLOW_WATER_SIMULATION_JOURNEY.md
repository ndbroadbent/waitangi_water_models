# Building a Shallow Water Simulation for Waitangi Estuary

## The Journey from "How Hard Can It Be?" to "Oh, That's Why CFD is a Field"

### Overview

This document chronicles the process of building a 2D shallow water simulation to model tidal flow in the Waitangi Estuary, New Zealand. What started as a seemingly straightforward task revealed the deep complexity of computational fluid dynamics and the importance of using proven numerical methods.

**Goal**: Simulate river water (green) flowing from Haruru Falls through the estuary and mixing with tidal water (blue) flowing in from the Bay of Islands.

**Spoiler**: It took multiple rewrites and a complete change of approach before achieving stable, physically plausible results.

---

## Phase 1: The Naive Approach

### Initial Implementation

Started with what seemed like a reasonable plan:
- Use JAX for GPU acceleration
- Implement the 2D shallow water equations in conservative form
- Use a MUSCL-Hancock scheme with HLL Riemann solver (sounds impressive!)
- Add tracer advection for river water visualization

```python
# The "proper" approach - conservative form
∂η/∂t + ∂(hu)/∂x + ∂(hv)/∂y = 0                    (continuity)
∂(hu)/∂t + ∂(hu²)/∂x + ∂(huv)/∂y = -gh∂η/∂x - τ_bx  (x-momentum)
∂(hv)/∂t + ∂(huv)/∂x + ∂(hv²)/∂y = -gh∂η/∂y - τ_by  (y-momentum)
```

### What Went Wrong

**Problem 1: NaN values everywhere**
```
Running simulation...
  14% - t=0.83h, tide=0.63m, max_depth=nanm
  28% - t=1.67h, tide=0.90m, max_depth=nanm
```

The HLL Riemann solver was producing NaN values when depths became very small. Wetting/drying is notoriously difficult to handle correctly.

**Problem 2: Unrealistic depths**
```
  max_depth=67.33m
  max_depth=82.42m
  max_depth=111.29m
```

Water was accumulating to absurd depths (100+ meters in an estuary that's only a few meters deep). Mass was being created from nowhere.

**Problem 3: Unrealistic velocities**
```
Max speed: 45.46 m/s
```

For context, 45 m/s is about 160 km/h - hurricane-force water velocities in a calm estuary!

### Attempted Fixes

1. **Switched from MUSCL-Hancock to simple upwind scheme** - Still unstable
2. **Added velocity limiting** - Helped symptoms, didn't fix root cause
3. **Added depth capping** - Band-aid, not a solution
4. **Implemented Flather boundary conditions** - Made it worse initially

---

## Phase 2: Researching the Literature

### What the Experts Do

Searched for established approaches and found key resources:

1. **[jostbr/shallow-water](https://github.com/jostbr/shallow-water)** - Simple, proven NumPy implementation
2. **[ROMS Wiki - Boundary Conditions](https://www.myroms.org/wiki/Boundary_Conditions)** - Flather radiation boundary
3. **[Veros Ocean Simulator](https://dionhaefner.github.io/2021/12/supercharged-high-resolution-ocean-simulation-with-jax/)** - JAX-based ocean modeling

### Key Insight: The Flather Boundary Condition

The standard approach for tidal boundaries is the **Flather condition**:

```
u = u_ext - √(g/h) × (η - η_ext)
```

This allows outgoing waves to radiate out while prescribing the tidal level at the boundary. However, implementing it correctly requires careful attention to:
- Sign conventions
- Depth calculations at boundaries
- Relaxation timescales

### The Real Problem

The fundamental issue wasn't the numerical scheme - it was the **problem setup**:

1. **Open boundary on entire eastern edge**: The simulation tried to fill/drain the entire Bay of Islands, not just the estuary
2. **No closed domain**: Water could flow everywhere, including onto dry land
3. **Boundary condition mismatch**: Setting a fixed water level at the boundary doesn't naturally create the correct inflow/outflow

---

## Phase 3: The Breakthrough - Closed Basin Approach

### A Simpler Question

Instead of asking "how do I simulate tidal flow?", the breakthrough came from asking:

> "What if we just draw an artificial wall around the basin and fill it with water at a fixed level?"

This is a **closed basin** approach:
1. Add a "wall" (high elevation) east of the bridge to close off the bay
2. Set a fixed water level source at that wall
3. Let physics fill the basin naturally

### Implementation

```python
# Create a wall to close off the bay
wall_col = 350
wall_height = 20.0  # meters

z_bed[:, wall_col:] = wall_height  # Everything east of wall is high ground
z_bed[:10, :] = wall_height        # North boundary
z_bed[-10:, :] = wall_height       # South boundary

# Source cells - where water enters at fixed level
source_mask[:, source_col:source_col+3] = z_bed[:, source_col:source_col+3] < tide_level

# In simulation loop:
eta[source_mask] = tide_level  # Enforce fixed water level at source
```

### The Simple Numerical Scheme That Works

Based on [jostbr/shallow-water](https://github.com/jostbr/shallow-water):

```python
# 1. Update velocities from surface gradient
u[:-1, :] -= g * dt * (eta[1:, :] - eta[:-1, :]) / dx
v[:, :-1] -= g * dt * (eta[:, 1:] - eta[:, :-1]) / dy

# 2. Apply friction
friction = 1.0 / (1.0 + dt * Cf * speed / h)
u *= friction
v *= friction

# 3. Upwind depth selection for mass flux
h_e = np.where(u > 0, h_left, h_right)

# 4. Update eta from flux divergence
eta = eta - dt * divergence(h * u, h * v)

# 5. Enforce: eta cannot go below z_bed
eta = np.maximum(eta, z_bed)
```

### Results

```
Expected area: 2.27 km²
Final area:    2.26 km²
Difference:    -0.014 km²

Overflow: 0.0195 km² (0.9%)
Underfill: 0.0335 km² (1.5%)
```

**Success!** The water fills to the expected level with minimal overflow.

---

## Lessons Learned

### 1. Start Simple, Add Complexity Later

The initial approach was too ambitious:
- MUSCL-Hancock reconstruction
- HLL Riemann solver
- Semi-implicit friction
- Open boundaries with Flather conditions

A simple first-order upwind scheme with explicit friction works perfectly well for this application.

### 2. The Problem Setup Matters More Than the Scheme

The biggest breakthrough wasn't a better numerical method - it was **closing the domain**. A simple scheme in a well-defined domain beats a sophisticated scheme in an ill-posed problem.

### 3. Boundary Conditions Are Hard

Open boundary conditions for tidal flow are genuinely difficult. The Flather condition is the standard, but requires:
- Correct sign conventions
- Appropriate relaxation timescales
- Well-defined exterior values

The closed basin approach sidesteps these issues entirely.

### 4. Test Incrementally

The successful approach was:
1. Fill a closed basin at constant water level
2. Verify it reaches the expected equilibrium
3. Only then add complexity (varying tide, river inflow, tracers)

### 5. Use Established Libraries

Resources like [jostbr/shallow-water](https://github.com/jostbr/shallow-water), [Veros](https://veros.readthedocs.io/), and the [ROMS Wiki](https://www.myroms.org/wiki/Boundary_Conditions) contain decades of accumulated knowledge. Don't reinvent the wheel.

---

## Technical Details

### Grid and Data

- **Source**: LINZ LiDAR DEM (1m resolution)
- **Domain**: Waitangi Estuary, Bay of Islands, New Zealand
- **Downsampled**: 8x (8m grid cells)
- **Grid size**: 319 × 410 cells

### Physical Parameters

- Gravity: 9.81 m/s²
- Manning's n: 0.035 (typical estuary)
- Tidal range: ~1.6m
- Mean water level: ~0.3m

### Numerical Parameters

- Timestep: ~0.29s (CFL limited)
- CFL factor: 0.2
- Velocity limit: 3 m/s

### Key Files

- `scripts/simple_tidal_test.py` - Working closed basin simulation
- `src/waitangi/simulation/shallow_water.py` - Original (complex) implementation
- `src/waitangi/data/elevation.py` - LiDAR data handling
- `reference_swe.py` - Downloaded from jostbr/shallow-water for reference

---

## Next Steps

1. **Vary tide level over time** to simulate rising/falling tide
2. **Add river source** at Haruru Falls with tracer
3. **Create animation** showing tidal flow and river mixing
4. **Add wind forcing** for kayak simulation
5. **Validate against** known tidal prism (3.08 million m³)

---

## References

1. jostbr/shallow-water: https://github.com/jostbr/shallow-water
2. ROMS Boundary Conditions: https://www.myroms.org/wiki/Boundary_Conditions
3. Veros Ocean Simulator: https://dionhaefner.github.io/2021/12/supercharged-high-resolution-ocean-simulation-with-jax/
4. Flather, R.A., 1976. A tidal model of the north-west European continental shelf. Mem. Soc. R. Sci. Liege 6, 141–164.
5. Carter and Merrifield (2007) - Open boundary conditions for regional tidal simulations

---

*Document created: December 2024*
*Project: Waitangi Water Models - Kayak Tidal Simulator*
