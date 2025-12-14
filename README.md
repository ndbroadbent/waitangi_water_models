# Waitangi River Kayak Tidal Simulator

GPU-accelerated simulator for predicting kayak motion in the Waitangi River system (Bay of Islands, New Zealand) as a function of tide, river discharge, wind, and paddling effort.

## Features

- **Tidal modeling**: Harmonic constituents with optional API refinement from NIWA
- **River discharge**: Live gauge data + rainfall-runoff forecasting
- **Wind effects**: MetService forecast integration with kayak drag model
- **GPU acceleration**: JAX-based computation for real-time particle advection
- **Visualization**: Trajectory plots, velocity profiles, GeoJSON export

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable Python package management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/ndbroadbent/waitangi_water_models.git
cd waitangi_water_models

# Sync dependencies (creates venv automatically)
uv sync

# For GPU support (NVIDIA CUDA)
uv sync --extra gpu

# For additional visualization tools
uv sync --extra viz
```

## Quick Start

```bash
# Show current conditions
uv run waitangi conditions

# Run a drift simulation (no paddling)
uv run waitangi simulate --duration 2.0 --plot

# Run with paddling
uv run waitangi simulate --duration 2.0 --paddle-speed 4.0 --direction upstream --plot

# View velocity profile along river
uv run waitangi profile

# Show conditions dashboard
uv run waitangi dashboard --hours 12

# Export river geometry to GeoJSON
uv run waitangi export-geojson river.geojson
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key settings:
- `DATA__METSERVICE_API_KEY`: MetService Point Forecast API key
- `DATA__LINZ_API_KEY`: LINZ Data Service API key
- `GPU_BACKEND`: `cpu`, `cuda`, or `metal`

## Python API

```python
import asyncio
from waitangi.simulation.runner import SimulationRunner
from waitangi.simulation.kayak import PaddlingProfile

async def main():
    # Initialize
    runner = SimulationRunner()
    await runner.initialize()

    # Get current conditions
    conditions = runner.get_current_conditions()
    print(f"Tide: {conditions['tide']['phase']}")
    print(f"Wind: {conditions['wind']['description']}")

    # Run simulation
    start_x, start_y = runner.mesh.chainage_to_point(500)  # 500m from mouth

    result = runner.run_single_kayak(
        start_x=start_x,
        start_y=start_y,
        duration_hours=2.0,
        paddling=PaddlingProfile.cruise_upstream(),
    )

    print(f"Distance traveled: {result.total_distance_m:.0f} m")
    print(f"Mean speed: {result.mean_speed_ms * 3.6:.1f} km/h")

asyncio.run(main())
```

## Architecture

```
src/waitangi/
├── core/           # Configuration, types, constants
├── data/           # Data ingestion (gauge, rainfall, tide, weather)
├── models/         # Physical models (geometry, tide, river, wind, velocity)
├── simulation/     # Simulation engine (kayak, particles, runner)
├── viz/            # Visualization (plots, maps, GeoJSON)
└── cli.py          # Command-line interface
```

### Key Models

1. **Tide Model**: Harmonic constituents (M2, S2, N2, K1, O1) with optional API calibration
2. **River Model**: Baseflow + exponential unit hydrograph for rainfall response
3. **Velocity Field**: Composition of tide, river, and optional eddy effects
4. **Particle System**: GPU-accelerated Lagrangian advection with RK2 integration

## Data Sources

- **River gauge**: Northland Regional Council Environmental Data Hub
- **Tide predictions**: NIWA or harmonic calculation
- **Weather forecast**: MetService Point Forecast API
- **Geometry**: LINZ hydrographic data / custom centerline

## Testing

```bash
# Run tests
uv run pytest

# With coverage
uv run pytest --cov=waitangi --cov-report=html
```

## License

MIT
