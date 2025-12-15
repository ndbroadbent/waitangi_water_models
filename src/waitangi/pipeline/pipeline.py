"""Pipeline orchestrator - coordinates all components."""

import sys
import time
from queue import Queue

from waitangi.pipeline.data import RenderConfig, SimulationConfig
from waitangi.pipeline.renderer import FrameRenderer
from waitangi.pipeline.simulation import SimulationEngine
from waitangi.pipeline.video_writer import VideoWriter


def _flushing_print(*args, **kwargs):
    """Print with immediate flush for real-time progress output."""
    print(*args, **kwargs)
    sys.stdout.flush()


class Pipeline:
    """Orchestrates the simulation, rendering, and video encoding pipeline.

    Architecture:
    ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
    │  SimulationEngine│────▶│  frame_queue     │────▶│  FrameRenderer  │
    │  (GPU/JAX)      │     │  (FrameData)     │     │  (matplotlib)   │
    └─────────────────┘     └──────────────────┘     └─────────────────┘
                                                              │
                                                              ▼
                                                     ┌──────────────────┐
                                                     │  render_queue    │
                                                     │  (RenderedFrame) │
                                                     └──────────────────┘
                                                              │
                                                              ▼
                                                     ┌─────────────────┐
                                                     │  VideoWriter    │
                                                     │  (ffmpeg)       │
                                                     └─────────────────┘
    """

    def __init__(
        self,
        sim_config: SimulationConfig | None = None,
        render_config: RenderConfig | None = None,
        num_render_workers: int = 4,
        frame_queue_size: int = 50,
        render_queue_size: int = 50,
        output_path: str | None = None,
        log_fn=_flushing_print,
    ):
        self.sim_config = sim_config or SimulationConfig()
        self.render_config = render_config or RenderConfig()
        self.num_render_workers = num_render_workers
        self.output_path = output_path
        self.log = log_fn

        # Queues for inter-component communication
        self.frame_queue: Queue = Queue(maxsize=frame_queue_size)
        self.render_queue: Queue = Queue(maxsize=render_queue_size)

        # Components (created during run)
        self.simulation: SimulationEngine | None = None
        self.renderer: FrameRenderer | None = None
        self.writer: VideoWriter | None = None

    def run(self):
        """Run the full pipeline.

        Starts all components, waits for completion, and reports statistics.
        """
        self.log("=" * 60)
        self.log("Waitangi Tidal Simulation Pipeline")
        self.log("=" * 60)
        self.log(f"River flow: {self.sim_config.river_flow} m³/s")
        self.log(f"Tracer diffusion: {self.sim_config.tracer_diffusion}")
        self.log(f"Render workers: {self.num_render_workers}")

        start_time = time.time()

        # Create simulation engine (this does setup/initialization)
        self.log("\n--- Initializing Simulation ---")
        self.simulation = SimulationEngine(
            config=self.sim_config,
            output_queue=self.frame_queue,
            log_fn=self.log,
        )

        # Need to run setup to get grid dimensions before creating renderer
        self.simulation._setup()

        # Create renderer
        self.log("\n--- Initializing Renderer ---")
        self.renderer = FrameRenderer(
            input_queue=self.frame_queue,
            output_queue=self.render_queue,
            config=self.render_config,
            grid_extent=self.simulation.grid_extent,
            grid_shape=self.simulation.grid_shape,
            downsample=self.simulation.downsample_factor,
            elevation_shape=self.simulation.elevation_data.data.shape,
            num_workers=self.num_render_workers,
            log_fn=self.log,
        )

        # Create video writer
        self.log("\n--- Initializing Video Writer ---")
        self.writer = VideoWriter(
            input_queue=self.render_queue,
            config=self.render_config,
            output_path=self.output_path,
            log_fn=self.log,
        )

        # Start all components (in reverse order so consumers are ready)
        self.log("\n--- Starting Pipeline ---")
        self.writer.start()
        self.renderer.start()

        # Run simulation in the main thread (uses GPU)
        # This is different from starting a thread - we run it directly
        self.log("\n--- Running Simulation ---")
        self.simulation._run_simulation()

        # Wait for pipeline to drain
        self.log("\n--- Waiting for pipeline to complete ---")
        self.renderer.join()
        self.writer.join()

        elapsed = time.time() - start_time
        self.log("\n" + "=" * 60)
        self.log("Pipeline Complete")
        self.log("=" * 60)
        self.log(f"Total time: {elapsed:.1f}s")
        self.log(f"Frames written: {self.writer.frames_written}")
        if self.writer.frames_written > 0:
            self.log(f"Average: {elapsed / self.writer.frames_written:.2f}s per frame")

        return self.output_path or self.writer.output_path

    def stop(self):
        """Stop all pipeline components gracefully."""
        if self.simulation:
            self.simulation.stop()
        if self.renderer:
            self.renderer.stop()
        if self.writer:
            self.writer.stop()


def run_pipeline(
    river_flow: float = 1.0,
    tracer_diffusion: float = 0.5,
    duration_hours: float | None = None,
    num_workers: int = 4,
    output_path: str | None = None,
) -> str:
    """Convenience function to run the full pipeline.

    Args:
        river_flow: River discharge in m³/s
        tracer_diffusion: Tracer diffusion coefficient (0 = no diffusion)
        duration_hours: Simulation duration (None = one tidal cycle)
        num_workers: Number of render workers
        output_path: Output video path (None = auto-generate)

    Returns:
        Path to output video file
    """
    sim_config = SimulationConfig(
        river_flow=river_flow,
        tracer_diffusion=tracer_diffusion,
        duration_hours=duration_hours,
    )

    pipeline = Pipeline(
        sim_config=sim_config,
        num_render_workers=num_workers,
        output_path=output_path,
    )

    return pipeline.run()
