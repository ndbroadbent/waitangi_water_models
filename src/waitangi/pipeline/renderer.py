"""Frame renderer using matplotlib.

Consumes FrameData from the simulation queue and produces PNG images
for the video writer.
"""

import io
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import TYPE_CHECKING

import contextily as ctx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.ndimage import zoom as scipy_zoom

from waitangi.pipeline.data import (
    END_OF_STREAM,
    FrameData,
    RenderConfig,
    RenderedFrame,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Configure module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Use non-interactive backend for thread safety
matplotlib.use("Agg")


def _blend_tracer_colors(river_tracer: "NDArray", ocean_tracer: "NDArray") -> "NDArray":
    """Blend colors based on dual tracer concentrations.

    Color scheme:
    - Yellow (1,1,0) = river water
    - Red (1,0,0) = ocean water
    - Orange (1,0.5,0) = mixed river+ocean
    - Blue (0.2,0.5,0.9) = neutral/untainted water
    """
    ny, nx = river_tracer.shape
    rgba = np.zeros((ny, nx, 4))

    blue = np.array([0.2, 0.5, 0.9])
    yellow = np.array([1.0, 0.95, 0.0])
    red = np.array([1.0, 0.1, 0.1])

    total = river_tracer + ocean_tracer
    total_safe = np.maximum(total, 1e-6)

    river_fraction = river_tracer / total_safe
    ocean_fraction = ocean_tracer / total_safe

    tracer_color = river_fraction[:, :, np.newaxis] * yellow + ocean_fraction[:, :, np.newaxis] * red

    blend_factor = np.minimum(total, 1.0)[:, :, np.newaxis]

    rgba[:, :, :3] = (1 - blend_factor) * blue + blend_factor * tracer_color
    rgba[:, :, 3] = 0.85

    return rgba


class FrameRenderer:
    """Multi-threaded frame renderer.

    Uses a thread pool to render frames in parallel while maintaining
    the correct frame order in the output queue.
    """

    def __init__(
        self,
        input_queue: Queue,
        output_queue: Queue,
        config: RenderConfig,
        grid_extent: tuple[float, float, float, float],
        grid_shape: tuple[int, int],
        downsample: int,
        elevation_shape: tuple[int, int],
        num_workers: int = 4,
        log_fn=None,
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.config = config
        self.extent = grid_extent
        self.ny, self.nx = grid_shape
        self.downsample = downsample
        self.elev_shape = elevation_shape
        self.num_workers = num_workers
        self.log = log_fn or logger.info

        self._stop_event = threading.Event()
        self._dispatcher_thread: threading.Thread | None = None
        self._executor: ThreadPoolExecutor | None = None

        # Pre-fetch basemap (shared across all workers)
        self._basemap_image = None
        self._basemap_extent = None
        self._fetch_basemap()

    def _fetch_basemap(self):
        """Pre-fetch the basemap image."""
        if not self.config.use_basemap:
            return

        self.log("Fetching basemap...")
        try:
            fig, ax = plt.subplots(figsize=(self.config.fig_width, self.config.fig_height))
            ax.set_xlim(self.extent[0], self.extent[1])
            ax.set_ylim(self.extent[2], self.extent[3])
            ctx.add_basemap(ax, crs="EPSG:2193", source=ctx.providers.Esri.WorldImagery, zoom=self.config.basemap_zoom)

            # Extract the basemap image from the axes
            for img in ax.images:
                self._basemap_image = img.get_array()
                self._basemap_extent = img.get_extent()
                break

            plt.close(fig)
            self.log("Basemap cached.")
        except Exception as e:
            self.log(f"Warning: basemap failed: {e}")
            self._basemap_image = None

    def _render_frame(self, frame: FrameData) -> RenderedFrame:
        """Render a single frame to PNG bytes."""
        cfg = self.config
        extent = [self.extent[0], self.extent[1], self.extent[2], self.extent[3]]

        fig, ax = plt.subplots(figsize=(cfg.fig_width, cfg.fig_height))
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        # Add basemap
        if self._basemap_image is not None:
            ax.imshow(
                self._basemap_image,
                extent=self._basemap_extent,
                origin="upper",
                zorder=0,
            )

        # Upsample for display
        h_full = scipy_zoom(frame.h, self.downsample, order=1)
        river_full = scipy_zoom(frame.river_tracer, self.downsample, order=1)
        ocean_full = scipy_zoom(frame.ocean_tracer, self.downsample, order=1)
        h_full = h_full[: self.elev_shape[0], : self.elev_shape[1]]
        river_full = river_full[: self.elev_shape[0], : self.elev_shape[1]]
        ocean_full = ocean_full[: self.elev_shape[0], : self.elev_shape[1]]

        # Create RGBA display
        wet = h_full > 0.05
        rgba = _blend_tracer_colors(river_full, ocean_full)
        rgba[:, :, 3] = np.where(wet, 0.85, 0.0)

        ax.imshow(rgba, origin="upper", extent=extent, zorder=1)

        # Add velocity arrows (quiver plot) - showing actual surface water speed
        # u,v are on staggered grid (cell faces), interpolate to cell centers
        u_center = np.zeros_like(frame.h)
        v_center = np.zeros_like(frame.h)
        # Average adjacent face values to get cell-center velocity
        u_center[1:, :] = (frame.u[:-1, :] + frame.u[1:, :]) / 2
        u_center[0, :] = frame.u[0, :]
        v_center[:, 1:] = (frame.v[:, :-1] + frame.v[:, 1:]) / 2
        v_center[:, 0] = frame.v[:, 0]

        skip = cfg.quiver_skip
        u_display = u_center[::skip, ::skip]
        v_display = v_center[::skip, ::skip]
        h_sub = frame.h[::skip, ::skip]

        x_coords = np.linspace(extent[0], extent[1], self.nx)[::skip]
        y_coords = np.linspace(extent[3], extent[2], self.ny)[::skip]
        X, Y = np.meshgrid(x_coords, y_coords)

        # Only show arrows where water depth > 5cm and speed > 1cm/s
        speed = np.sqrt(u_display**2 + v_display**2)
        mask = (h_sub > 0.05) & (speed > 0.01)

        if mask.sum() > 0:
            # Coordinate system mapping for quiver arrows:
            #
            # SIMULATION GRID (row-major numpy array):
            #   - Array indexing: arr[row, col] where row increases downward, col increases rightward
            #   - u: velocity between rows (u[i,:] is flux from row i to row i+1)
            #        positive u = flow toward increasing row index = SOUTHWARD (down in image)
            #   - v: velocity between columns (v[:,j] is flux from col j to col j+1)
            #        positive v = flow toward increasing col index = EASTWARD (right in image)
            #
            # PLOT COORDINATES (NZTM/EPSG:2193):
            #   - x-axis: Easting (increases rightward = East)
            #   - y-axis: Northing (increases upward = North)
            #   - y_coords are reversed: linspace(ymax, ymin) so row 0 is at top (North)
            #
            # MAPPING:
            #   - Simulation v (eastward) → Plot x-component (no change needed)
            #   - Simulation u (southward) → Plot y-component (negate: south = -y)
            #
            # Result: quiver(X, Y, v, -u) shows arrows in correct geographic direction
            ax.quiver(
                X[mask],
                Y[mask],
                v_display[mask],
                -u_display[mask],
                color="white",
                alpha=cfg.quiver_alpha,
                scale=cfg.quiver_scale,
                width=0.002,
                headwidth=4,
                headlength=5,
                zorder=3,
            )

        # Plot gauge markers
        marker_colors = ["white", "cyan", "yellow"]
        for gi, g in enumerate(frame.gauges):
            ax.plot(
                g.x,
                g.y,
                "o",
                color=marker_colors[gi % len(marker_colors)],
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=1.5,
            )
            ax.annotate(
                g.name,
                (g.x, g.y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                color="white",
                path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground="black")],
            )

        # Draw kayak
        heading_rad = np.radians(frame.kayak.heading)
        scale = cfg.kayak_scale
        kayak = frame.kayak
        front = (kayak.x + scale * np.cos(heading_rad), kayak.y + scale * np.sin(heading_rad))
        back = (kayak.x - scale * 0.5 * np.cos(heading_rad), kayak.y - scale * 0.5 * np.sin(heading_rad))
        right = (
            kayak.x + scale * 0.3 * np.cos(heading_rad - np.pi / 2),
            kayak.y + scale * 0.3 * np.sin(heading_rad - np.pi / 2),
        )
        left = (
            kayak.x + scale * 0.3 * np.cos(heading_rad + np.pi / 2),
            kayak.y + scale * 0.3 * np.sin(heading_rad + np.pi / 2),
        )
        kayak_verts_x = [front[0], right[0], back[0], left[0], front[0]]
        kayak_verts_y = [front[1], right[1], back[1], left[1], front[1]]
        ax.fill(kayak_verts_x, kayak_verts_y, color="lime", edgecolor="black", linewidth=2, zorder=10)

        # Legend
        legend_elements = [
            Patch(facecolor=(1.0, 0.95, 0.0), edgecolor="black", label="River (yellow)"),
            Patch(facecolor=(1.0, 0.1, 0.1), edgecolor="black", label="Ocean (red)"),
            Patch(facecolor=(1.0, 0.5, 0.05), edgecolor="black", label="Mixed (orange)"),
            Patch(facecolor=(0.2, 0.5, 0.9), edgecolor="black", label="Neutral (blue)"),
            Patch(facecolor="lime", edgecolor="black", label="Kayak"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

        # Title
        hours = int(frame.simulation_time // 3600)
        mins = int((frame.simulation_time % 3600) // 60)
        ax.set_title(
            f"Waitangi Estuary - Tidal Simulation\n"
            f"Time: {hours}h {mins:02d}m | Tide: {frame.tide_level:+.2f}m | River: {frame.river_flow:.1f} m³/s\n"
            f"Flooded: {frame.wet_area_km2:.2f} km² | Frame {frame.frame_number+1}/{frame.total_frames}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")

        # Gauge data text box
        gauge_text = "Flow Gauges:\n"
        for gd in frame.gauges:
            gauge_text += f"{gd.name:12s} h={gd.depth:.2f}m  v={gd.speed:.2f}m/s {gd.direction:>2s}  Q={gd.flow:.1f}m³/s\n"

        ax.text(
            0.98,
            0.02,
            gauge_text.strip(),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            fontfamily="monospace",
            multialignment="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )

        plt.tight_layout()

        # Save to PNG bytes
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=cfg.dpi, bbox_inches="tight", facecolor="white")
        buf.seek(0)
        png_data = buf.read()
        plt.close(fig)

        return RenderedFrame(frame_number=frame.frame_number, png_data=png_data)

    def _dispatcher_loop(self):
        """Main dispatcher loop - submits frames to thread pool."""
        while not self._stop_event.is_set():
            try:
                item = self.input_queue.get(timeout=0.1)
            except Exception:
                continue

            if item is END_OF_STREAM:
                self.output_queue.put(END_OF_STREAM)
                break

            # Submit to thread pool
            future = self._executor.submit(self._render_frame, item)
            # Get result and put in output queue
            # Note: This blocks, maintaining order
            # For true parallelism with ordering, we'd need a more complex approach
            try:
                rendered = future.result()
                self.output_queue.put(rendered)
            except Exception as e:
                self.log(f"Error rendering frame {item.frame_number}: {e}")

    def start(self):
        """Start the renderer workers."""
        self._stop_event.clear()
        self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self._dispatcher_thread = threading.Thread(target=self._dispatcher_loop, daemon=True)
        self._dispatcher_thread.start()

    def stop(self):
        """Stop the renderer gracefully."""
        self._stop_event.set()
        if self._dispatcher_thread:
            self._dispatcher_thread.join(timeout=5.0)
        if self._executor:
            self._executor.shutdown(wait=False)

    def join(self):
        """Wait for renderer to complete."""
        if self._dispatcher_thread:
            self._dispatcher_thread.join()
        if self._executor:
            self._executor.shutdown(wait=True)
