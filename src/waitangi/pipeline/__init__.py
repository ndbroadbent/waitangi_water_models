"""Pipelined tidal simulation with parallel GPU simulation and CPU rendering.

Architecture:
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  GPU Simulation │────▶│  Frame Queue     │────▶│  Render Workers │
│  (JAX Metal)    │     │  (state data)    │     │  (matplotlib)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
                                                 ┌──────────────────┐
                                                 │  Image Queue     │
                                                 │  (ordered dict)  │
                                                 └──────────────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │  FFmpeg Writer  │
                                                 │  (frame order)  │
                                                 └─────────────────┘

Benefits:
- GPU runs at full speed without waiting for matplotlib
- Multiple CPU threads render frames in parallel
- FFmpeg writes frames in order for correct video
"""

from waitangi.pipeline.data import FrameData, SimulationConfig, RenderConfig
from waitangi.pipeline.simulation import SimulationEngine
from waitangi.pipeline.renderer import FrameRenderer
from waitangi.pipeline.video_writer import VideoWriter
from waitangi.pipeline.pipeline import Pipeline

__all__ = [
    "FrameData",
    "SimulationConfig",
    "RenderConfig",
    "SimulationEngine",
    "FrameRenderer",
    "VideoWriter",
    "Pipeline",
]
