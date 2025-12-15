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

import sys

# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

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

    # Debug mode: run simulation only (no rendering/video), print tracer stats
    if args.debug_frame is not None:
        from waitangi.pipeline.simulation import SimulationEngine
        from waitangi.pipeline.data import END_OF_STREAM
        from queue import Queue
        import numpy as np

        print(f"=== DEBUG MODE: Running until frame {args.debug_frame} ===")
        print(f"River flow: {sim_config.river_flow} m³/s")
        print(f"Tracer diffusion: {sim_config.tracer_diffusion}")
        print(f"Skip equilibrium: {sim_config.skip_equilibrium}")
        print(f"Fixed tide: {sim_config.fixed_tide}")

        frame_queue = Queue(maxsize=100)

        engine = SimulationEngine(
            config=sim_config,
            output_queue=frame_queue,
            log_fn=print,
        )
        engine._setup()
        engine._run_simulation(stop_at_frame=args.debug_frame)

        # Drain queue and get frames
        frames = []
        while True:
            item = frame_queue.get()
            if item is END_OF_STREAM:
                break
            frames.append(item)

        if not frames:
            print("ERROR: No frames produced!")
            return

        frame = frames[-1]
        print(f"\n=== TRACER STATS AT FRAME {frame.frame_number} ===")
        h = frame.h
        river_tracer = frame.river_tracer
        ocean_tracer = frame.ocean_tracer

        wet = h > 0.05
        wet_count = np.sum(wet)

        traced = (ocean_tracer > 0.5) | (river_tracer > 0.5)
        traced_wet = traced & wet
        untraced_wet = ~traced & wet

        traced_count = np.sum(traced_wet)
        untraced_count = np.sum(untraced_wet)

        print(f"Wet cells: {wet_count}")
        print(f"Traced wet cells (tracer > 0.5): {traced_count}")
        print(f"Untraced wet cells (tracer <= 0.5): {untraced_count}")
        print(f"Ratio traced/untraced: {traced_count}/{untraced_count}")
        if wet_count > 0:
            print(f"Percent traced: {100 * traced_count / wet_count:.1f}%")

        print(f"\n=== DETAILED TRACER VALUES ===")
        ocean_wet = ocean_tracer[wet]
        river_wet = river_tracer[wet]
        print(f"Ocean tracer in wet cells: min={ocean_wet.min():.3f}, max={ocean_wet.max():.3f}, mean={ocean_wet.mean():.3f}")
        print(f"River tracer in wet cells: min={river_wet.min():.3f}, max={river_wet.max():.3f}, mean={river_wet.mean():.3f}")

        print(f"\nOcean tracer distribution in wet cells:")
        for thresh in [0.0, 0.1, 0.5, 0.9, 1.0]:
            count = np.sum(ocean_wet >= thresh)
            print(f"  >= {thresh}: {count} ({100*count/len(ocean_wet):.1f}%)")

        # Find where untraced cells are located
        untraced_rows, untraced_cols = np.where(untraced_wet)
        if len(untraced_rows) > 0:
            print(f"\n=== UNTRACED CELL LOCATIONS ===")
            print(f"Row range: {untraced_rows.min()} - {untraced_rows.max()}")
            print(f"Col range: {untraced_cols.min()} - {untraced_cols.max()}")
            print(f"Grid shape: {h.shape}")
            # Show first 10 untraced cell locations and their tracer values
            print(f"\nFirst 10 untraced cells (row, col, ocean_tracer, h):")
            for i in range(min(10, len(untraced_rows))):
                r, c = untraced_rows[i], untraced_cols[i]
                print(f"  ({r}, {c}): ocean={ocean_tracer[r,c]:.4f}, h={h[r,c]:.4f}")

        return

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
