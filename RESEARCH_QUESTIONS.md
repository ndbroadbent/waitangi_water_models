# Waitangi Estuary Research Questions

Interesting questions to explore with the simulation model.

## Answered

### At what elevation do mangroves establish?
**Answer: 0.0m (mean sea level)**

Discovered empirically using the mangrove finder video - frame 11 at exactly 0.0m water level is where sandbars disappear and only mangrove areas remain exposed.

This aligns with biology:
- Below 0m: Flooded >50% of time - too wet for root respiration
- Above 0m: Exposed >50% of time - allows pneumatophores to breathe, seedlings to anchor

The 0m threshold matches the NZVD2016 vertical datum (mean sea level), making it a natural biological boundary.

**Files:** `scripts/mangrove_finder.py`, `scripts/generate_mangrove_polygon.py`

---

## Open Questions

### At what river flow does rainfall overtake tidal influence?
During the late September flood, flow hit 15.8 m³/s (50x normal baseflow). At what threshold does river inflow dominate the estuary, forcing all water outward regardless of tide?

**Hypothesis:** When river inflow exceeds the tidal exchange rate, the estuary becomes river-dominated.
- Tidal prism: ~3.08 million m³ exchanged over ~6 hours = ~143 m³/s average tidal flow
- Even at 15 m³/s flood, tides may still dominate overall volume
- But velocity at constrictions (bridge) could reverse - worth simulating

**To investigate:**
- Model estuary response at different river inflow rates (0.3, 1, 5, 10, 15 m³/s)
- Find the crossover point where outflow is continuous through a full tidal cycle

### How fast does flood water flow under the bridge?
During a 15 m³/s flood event with 2.8m water level upstream:
- Does it create rapids under the bridge?
- How does this interact with incoming tide?
- Is there a dangerous standing wave or turbulence?

### What happens to water level in the estuary during floods?
Does the estuary:
- Fill up and match the upstream level?
- Stay at tidal level while river rushes through?
- Create a gradient from Haruru Falls to the bay?

### Can we predict kayak conditions from gauge data?
Using the Waimate North Rd gauge:
- What flow/stage thresholds indicate "too fast" for paddling upstream?
- What indicates good drift conditions for a lazy downstream paddle?
- Correlation between gauge readings and actual estuary current speeds?

### How does the mangrove zone affect flood dynamics?
Mangroves provide friction and slow water flow. During floods:
- How much do they dampen the flood wave?
- Do they cause water to back up?
- What's the velocity difference between channels and mangrove areas?

---

## Data Sources

- **Elevation:** LINZ LiDAR DEM 1m (sheet AV29)
- **River flow/level:** NRC Hilltop API - "Waitangi at Waimate North Rd"
- **Tidal predictions:** To be added (LINZ or NIWA tide tables)
- **Ground truth:** Reference points from local knowledge

## Key Numbers

| Metric | Value | Notes |
|--------|-------|-------|
| Tidal prism | 3.08 million m³ | Volume exchanged low→high tide |
| Mangrove area | 1.406 km² | Elevation 0.0m to 1.1m |
| Low tide level | -0.5m | Typical low |
| High tide level | +1.1m | Typical high |
| Normal river flow | 0.3-1.0 m³/s | Baseflow conditions |
| Flood flow (Sept 2025) | 15.8 m³/s | Peak observed |
| Normal stage | ~750mm | At Waimate North Rd |
| Flood stage (Sept 2025) | 2829mm | Peak observed |
