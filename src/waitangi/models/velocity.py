"""Water velocity field composition.

Combines tide, river discharge, and optional eddy effects into
a unified velocity field for particle advection.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import jax.numpy as jnp
from jax import Array, jit

from waitangi.models.geometry import Mesh
from waitangi.models.river import RiverDischargeModel
from waitangi.models.tide import TideModel


@dataclass
class VelocityField:
    """Composed water velocity field.

    At each position and time:
    v_water(x,y,t) = v_tide(x,t) - v_river(x,t) + v_eddy(x,y)

    Where:
    - v_tide: Tidal current (positive = flood/upstream, decays inland)
    - v_river: River flow (positive = downstream, decays from source)
    - v_eddy: Optional turbulent/recirculating zones

    Sign convention:
    - Positive along-river = upstream (flood direction)
    - River flow is always downstream, so subtracted
    """

    mesh: Mesh
    tide_model: TideModel
    river_model: RiverDischargeModel

    # Optional eddy field function
    eddy_fn: Callable[[Array, Array], tuple[Array, Array]] | None = None

    def get_velocity_at_point(
        self,
        x: float,
        y: float,
        timestamp: datetime,
    ) -> tuple[float, float]:
        """Get water velocity at a single point.

        Args:
            x: X coordinate (NZTM).
            y: Y coordinate (NZTM).
            timestamp: Time for calculation.

        Returns:
            Tuple of (u, v) velocity in m/s.
        """
        # Get distance along river
        chainage = self.mesh.point_to_chainage(x, y)

        # Get river direction at this point (tangent to centerline)
        dir_x, dir_y = self._get_river_direction(chainage)

        # Tidal velocity (positive = flood = upstream)
        v_tide = float(self.tide_model.get_velocity_field(timestamp, chainage))

        # River velocity (positive = downstream)
        v_river = float(self.river_model.get_velocity_field(timestamp, chainage))

        # Combined along-river velocity
        # Tide pushes upstream (+), river flows downstream (-)
        v_along = v_tide - v_river

        # Convert to cartesian
        u = v_along * dir_x
        v = v_along * dir_y

        # Add eddy component if defined
        if self.eddy_fn is not None:
            u_eddy, v_eddy = self.eddy_fn(jnp.array([x]), jnp.array([y]))
            u += float(u_eddy[0])
            v += float(v_eddy[0])

        return u, v

    def get_velocity_field_at_nodes(
        self,
        timestamp: datetime,
    ) -> tuple[Array, Array]:
        """Get velocity at all mesh nodes.

        Args:
            timestamp: Time for calculation.

        Returns:
            Tuple of (u, v) arrays at each node.
        """
        # Calculate chainage for all nodes
        chainages = vmap(self.mesh.point_to_chainage)(
            self.mesh.node_x, self.mesh.node_y
        )

        # Get tidal velocities
        v_tide = self.tide_model.get_velocity_field(timestamp, chainages)

        # Get river velocities
        v_river = self.river_model.get_velocity_field(timestamp, chainages)

        # Combined along-river velocity
        v_along = v_tide - v_river

        # Get river directions at each node
        dir_x, dir_y = self._get_river_directions_batch(chainages)

        # Convert to cartesian
        u = v_along * dir_x
        v = v_along * dir_y

        # Add eddy components
        if self.eddy_fn is not None:
            u_eddy, v_eddy = self.eddy_fn(self.mesh.node_x, self.mesh.node_y)
            u = u + u_eddy
            v = v + v_eddy

        return u, v

    def get_velocity_at_particles(
        self,
        x: Array,
        y: Array,
        timestamp: datetime,
    ) -> tuple[Array, Array]:
        """Get velocity at particle positions.

        Optimized for GPU execution with batched particles.

        Args:
            x: Array of particle x coordinates.
            y: Array of particle y coordinates.
            timestamp: Time for calculation.

        Returns:
            Tuple of (u, v) velocity arrays.
        """
        # Calculate chainage for all particles
        chainages = self._batch_chainage(x, y)

        # Get tidal velocities
        v_tide = self.tide_model.get_velocity_field(timestamp, chainages)

        # Get river velocities
        v_river = self.river_model.get_velocity_field(timestamp, chainages)

        # Combined along-river velocity
        v_along = v_tide - v_river

        # Get river directions
        dir_x, dir_y = self._get_river_directions_batch(chainages)

        # Convert to cartesian
        u = v_along * dir_x
        v = v_along * dir_y

        # Add eddy components
        if self.eddy_fn is not None:
            u_eddy, v_eddy = self.eddy_fn(x, y)
            u = u + u_eddy
            v = v + v_eddy

        return u, v

    def get_cancellation_zone(self, timestamp: datetime) -> float:
        """Find the chainage where tidal and river flows cancel.

        This is the "stagnation line" where water is nearly stationary.

        Args:
            timestamp: Time for calculation.

        Returns:
            Chainage in meters where velocity is approximately zero.
        """
        # Search along river
        chainages = jnp.linspace(0, self.mesh.river_length, 100)

        v_tide = self.tide_model.get_velocity_field(timestamp, chainages)
        v_river = self.river_model.get_velocity_field(timestamp, chainages)

        v_net = v_tide - v_river

        # Find where velocity crosses zero
        sign_changes = jnp.where(v_net[:-1] * v_net[1:] < 0)[0]

        if len(sign_changes) > 0:
            idx = int(sign_changes[0])
            # Linear interpolation to find exact crossing
            v0, v1 = float(v_net[idx]), float(v_net[idx + 1])
            c0, c1 = float(chainages[idx]), float(chainages[idx + 1])
            alpha = -v0 / (v1 - v0) if abs(v1 - v0) > 1e-10 else 0.5
            return c0 + alpha * (c1 - c0)

        # No crossing found - return boundary
        if float(v_net[0]) > 0:
            return 0.0  # Flood dominant at mouth
        return float(self.mesh.river_length)  # Ebb dominant to upstream

    def get_velocity_profile(
        self,
        timestamp: datetime,
        n_points: int = 50,
    ) -> dict:
        """Get velocity profile along river centerline.

        Args:
            timestamp: Time for calculation.
            n_points: Number of points along river.

        Returns:
            Dictionary with profile data.
        """
        chainages = jnp.linspace(0, self.mesh.river_length, n_points)

        v_tide = self.tide_model.get_velocity_field(timestamp, chainages)
        v_river = self.river_model.get_velocity_field(timestamp, chainages)
        v_net = v_tide - v_river

        return {
            "chainage_m": chainages,
            "v_tide_ms": v_tide,
            "v_river_ms": v_river,
            "v_net_ms": v_net,
            "cancellation_m": self.get_cancellation_zone(timestamp),
        }

    def _get_river_direction(self, chainage: float) -> tuple[float, float]:
        """Get unit vector in upstream direction at given chainage."""
        # Get centerline segment
        idx = jnp.searchsorted(self.mesh.river_chainage, chainage)
        idx = jnp.clip(idx, 1, self.mesh.n_centerline_points - 1)

        x0 = self.mesh.river_centerline_x[idx - 1]
        y0 = self.mesh.river_centerline_y[idx - 1]
        x1 = self.mesh.river_centerline_x[idx]
        y1 = self.mesh.river_centerline_y[idx]

        dx = x1 - x0
        dy = y1 - y0
        length = jnp.sqrt(dx**2 + dy**2) + 1e-10

        # Upstream direction (from mouth towards source)
        return float(dx / length), float(dy / length)

    def _get_river_directions_batch(
        self, chainages: Array
    ) -> tuple[Array, Array]:
        """Get river direction vectors for multiple chainages."""
        # Vectorized version of _get_river_direction
        indices = jnp.searchsorted(self.mesh.river_chainage, chainages)
        indices = jnp.clip(indices, 1, self.mesh.n_centerline_points - 1)

        x0 = self.mesh.river_centerline_x[indices - 1]
        y0 = self.mesh.river_centerline_y[indices - 1]
        x1 = self.mesh.river_centerline_x[indices]
        y1 = self.mesh.river_centerline_y[indices]

        dx = x1 - x0
        dy = y1 - y0
        lengths = jnp.sqrt(dx**2 + dy**2) + 1e-10

        return dx / lengths, dy / lengths

    def _batch_chainage(self, x: Array, y: Array) -> Array:
        """Calculate chainage for batch of points.

        Uses nearest centerline point (could be improved with
        projection onto centerline segments).
        """
        # Compute distances to all centerline points for all particles
        # Shape: (n_particles, n_centerline_points)
        dx = x[:, None] - self.mesh.river_centerline_x[None, :]
        dy = y[:, None] - self.mesh.river_centerline_y[None, :]
        distances = jnp.sqrt(dx**2 + dy**2)

        # Find nearest centerline point for each particle
        nearest_indices = jnp.argmin(distances, axis=1)

        # Return chainage at nearest points
        return self.mesh.river_chainage[nearest_indices]


def create_eddy_field(
    mesh: Mesh,
    eddy_locations: list[tuple[float, float]] | None = None,
    eddy_strength: float = 0.2,
    eddy_radius: float = 30.0,
) -> Callable[[Array, Array], tuple[Array, Array]]:
    """Create an eddy field function for recirculating zones.

    Args:
        mesh: Mesh geometry.
        eddy_locations: List of (x, y) eddy center locations.
                       If None, auto-generates based on bends.
        eddy_strength: Maximum eddy velocity (m/s).
        eddy_radius: Characteristic eddy radius (m).

    Returns:
        Function (x, y) -> (u_eddy, v_eddy).
    """
    if eddy_locations is None:
        eddy_locations = _detect_bend_locations(mesh)

    eddy_centers = jnp.array(eddy_locations)
    n_eddies = len(eddy_locations)

    if n_eddies == 0:
        # No eddies - return zero field
        def zero_eddy(x: Array, y: Array) -> tuple[Array, Array]:
            return jnp.zeros_like(x), jnp.zeros_like(y)
        return zero_eddy

    @jit
    def eddy_fn(x: Array, y: Array) -> tuple[Array, Array]:
        """Compute eddy velocities at given positions."""
        u_total = jnp.zeros_like(x)
        v_total = jnp.zeros_like(y)

        for i in range(n_eddies):
            cx, cy = eddy_centers[i]

            # Distance from eddy center
            dx = x - cx
            dy = y - cy
            r = jnp.sqrt(dx**2 + dy**2) + 1e-10

            # Rankine vortex profile
            # Inside core: v ~ r
            # Outside core: v ~ 1/r
            v_theta = eddy_strength * jnp.where(
                r < eddy_radius,
                r / eddy_radius,
                eddy_radius / r,
            ) * jnp.exp(-r / (3 * eddy_radius))

            # Convert to cartesian (perpendicular to radius)
            u_eddy = -v_theta * dy / r
            v_eddy = v_theta * dx / r

            u_total = u_total + u_eddy
            v_total = v_total + v_eddy

        return u_total, v_total

    return eddy_fn


def _detect_bend_locations(mesh: Mesh) -> list[tuple[float, float]]:
    """Detect river bends where eddies might form.

    Looks for high curvature points in the centerline.
    """
    if mesh.n_centerline_points < 5:
        return []

    x = mesh.river_centerline_x
    y = mesh.river_centerline_y

    # Calculate curvature using second derivative
    dx = jnp.diff(x)
    dy = jnp.diff(y)
    ddx = jnp.diff(dx)
    ddy = jnp.diff(dy)

    # Curvature = |x'y'' - y'x''| / (x'^2 + y'^2)^1.5
    curvature = jnp.abs(dx[:-1] * ddy - dy[:-1] * ddx) / (
        (dx[:-1]**2 + dy[:-1]**2) ** 1.5 + 1e-10
    )

    # Find high curvature points (above median)
    threshold = float(jnp.median(curvature)) * 2

    bend_indices = jnp.where(curvature > threshold)[0] + 1  # +1 for offset

    locations = []
    for idx in bend_indices:
        idx = int(idx)
        if 0 < idx < mesh.n_centerline_points - 1:
            locations.append((float(x[idx]), float(y[idx])))

    return locations
