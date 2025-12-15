#!/usr/bin/env python3
"""CLI entry point for the pipelined tidal simulation.

This uses a parallel pipeline architecture:
- GPU thread runs JAX simulation at full speed
- Multiple CPU threads render matplotlib frames in parallel
- FFmpeg thread writes frames in order to video

Usage:
    uv run python scripts/run_pipeline.py --duration 1 --river-flow 1.0
    uv run python scripts/run_pipeline.py --duration 12 --river-flow 0.0 --workers 8
"""

import argparse

from waitangi.pipeline import Pipeline, RenderConfig, SimulationConfig


def main():
    parser = argparse.ArgumentParser(
        description="Waitangi Estuary Tidal Simulation (Pipelined)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--river-flow",
        type=float,
        default=1.0,
        help="River flow rate in m³/s (use 15.0 for heavy rain)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Simulation duration in hours (default: full tidal cycle ~12.4h)",
    )
    parser.add_argument(
        "--no-diffusion",
        action="store_true",
        help="Disable tracer diffusion (permanent dye tracking)",
    )
    parser.add_argument(
        "--diffusion",
        type=float,
        default=0.5,
        help="Tracer diffusion coefficient",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of render worker threads",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path (default: auto-generated timestamp)",
    )
    parser.add_argument(
        "--frame-queue-size",
        type=int,
        default=50,
        help="Size of frame data queue (simulation -> renderer)",
    )
    parser.add_argument(
        "--render-queue-size",
        type=int,
        default=50,
        help="Size of rendered frame queue (renderer -> writer)",
    )
    parser.add_argument(
        "--skip-equilibrium",
        action="store_true",
        help="Skip the initial equilibration phase",
    )
    parser.add_argument(
        "--fixed-tide",
        type=float,
        default=None,
        help="Fixed tide level (disables tidal cycle)",
    )
    parser.add_argument(
        "--debug-frame",
        type=int,
        default=None,
        help="Run simulation until this frame and print tracer stats (no video)",
    )

    args = parser.parse_args()

    tracer_diffusion = 0.0 if args.no_diffusion else args.diffusion

    sim_config = SimulationConfig(
        river_flow=args.river_flow,
        tracer_diffusion=tracer_diffusion,
        duration_hours=args.duration,
        skip_equilibrium=args.skip_equilibrium,
        fixed_tide=args.fixed_tide,
    )

    render_config = RenderConfig()

    pipeline = Pipeline(
        sim_config=sim_config,
        render_config=render_config,
        num_render_workers=args.workers,
        frame_queue_size=args.frame_queue_size,
        render_queue_size=args.render_queue_size,
        output_path=args.output,
    )

    try:
        output_path = pipeline.run()
        print(f"\nOutput: {output_path}")
    except KeyboardInterrupt:
        print("\nInterrupted - stopping pipeline...")
        pipeline.stop()


if __name__ == "__main__":
    main()
