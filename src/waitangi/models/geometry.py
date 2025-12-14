"""Geometry and mesh handling for the simulation domain.

Uses unstructured triangular meshes for GPU-friendly computation.
Data layout follows structure-of-arrays (SoA) for GPU efficiency.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import jax.numpy as jnp
import numpy as np
import triangle
from jax import Array
from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon

from waitangi.core.config import WaitangiLocation
from waitangi.core.types import FloatArray


@dataclass
class Mesh:
    """Unstructured triangular mesh for the simulation domain.

    All arrays are stored in structure-of-arrays format for GPU efficiency.
    Coordinates are in NZTM2000 (EPSG:2193) for metric calculations.
    """

    # Node data (N nodes)
    node_x: Array  # (N,) x coordinates in meters
    node_y: Array  # (N,) y coordinates in meters
    node_depth: Array  # (N,) depth at each node in meters (positive down)

    # Element (triangle) data (M triangles)
    tri_nodes: Array  # (M, 3) node indices for each triangle

    # Precomputed values for interpolation
    tri_centroids_x: Array  # (M,) x coordinate of triangle centroids
    tri_centroids_y: Array  # (M,) y coordinate of triangle centroids
    tri_areas: Array  # (M,) area of each triangle

    # River geometry (for along-river calculations)
    river_centerline_x: Array  # (K,) centerline x coordinates
    river_centerline_y: Array  # (K,) centerline y coordinates
    river_chainage: Array  # (K,) distance along river from mouth

    # Boundary markers
    mouth_node_indices: Array  # Nodes at river mouth (ocean boundary)
    upstream_node_indices: Array  # Nodes at upstream boundary

    @property
    def n_nodes(self) -> int:
        """Number of nodes in mesh."""
        return int(self.node_x.shape[0])

    @property
    def n_triangles(self) -> int:
        """Number of triangles in mesh."""
        return int(self.tri_nodes.shape[0])

    @property
    def n_centerline_points(self) -> int:
        """Number of points in river centerline."""
        return int(self.river_centerline_x.shape[0])

    @property
    def river_length(self) -> float:
        """Total river length in meters."""
        return float(self.river_chainage[-1])

    def to_numpy(self) -> dict[str, np.ndarray]:
        """Export all arrays to numpy for serialization."""
        return {
            "node_x": np.asarray(self.node_x),
            "node_y": np.asarray(self.node_y),
            "node_depth": np.asarray(self.node_depth),
            "tri_nodes": np.asarray(self.tri_nodes),
            "tri_centroids_x": np.asarray(self.tri_centroids_x),
            "tri_centroids_y": np.asarray(self.tri_centroids_y),
            "tri_areas": np.asarray(self.tri_areas),
            "river_centerline_x": np.asarray(self.river_centerline_x),
            "river_centerline_y": np.asarray(self.river_centerline_y),
            "river_chainage": np.asarray(self.river_chainage),
            "mouth_node_indices": np.asarray(self.mouth_node_indices),
            "upstream_node_indices": np.asarray(self.upstream_node_indices),
        }

    def save(self, path: Path) -> None:
        """Save mesh to NPZ file."""
        np.savez_compressed(path, **self.to_numpy())

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load mesh from NPZ file."""
        data = np.load(path)
        return cls(
            node_x=jnp.asarray(data["node_x"]),
            node_y=jnp.asarray(data["node_y"]),
            node_depth=jnp.asarray(data["node_depth"]),
            tri_nodes=jnp.asarray(data["tri_nodes"]),
            tri_centroids_x=jnp.asarray(data["tri_centroids_x"]),
            tri_centroids_y=jnp.asarray(data["tri_centroids_y"]),
            tri_areas=jnp.asarray(data["tri_areas"]),
            river_centerline_x=jnp.asarray(data["river_centerline_x"]),
            river_centerline_y=jnp.asarray(data["river_centerline_y"]),
            river_chainage=jnp.asarray(data["river_chainage"]),
            mouth_node_indices=jnp.asarray(data["mouth_node_indices"]),
            upstream_node_indices=jnp.asarray(data["upstream_node_indices"]),
        )

    def point_to_chainage(self, x: float, y: float) -> float:
        """Convert (x, y) position to distance along river from mouth.

        Uses nearest centerline point.
        """
        distances = jnp.sqrt(
            (self.river_centerline_x - x) ** 2 + (self.river_centerline_y - y) ** 2
        )
        nearest_idx = jnp.argmin(distances)
        return float(self.river_chainage[nearest_idx])

    def chainage_to_point(self, chainage: float) -> tuple[float, float]:
        """Convert distance along river to (x, y) position.

        Interpolates along centerline.
        """
        idx = jnp.searchsorted(self.river_chainage, chainage)
        idx = jnp.clip(idx, 1, self.n_centerline_points - 1)

        # Linear interpolation
        c0 = self.river_chainage[idx - 1]
        c1 = self.river_chainage[idx]
        alpha = (chainage - c0) / (c1 - c0 + 1e-10)

        x = self.river_centerline_x[idx - 1] + alpha * (
            self.river_centerline_x[idx] - self.river_centerline_x[idx - 1]
        )
        y = self.river_centerline_y[idx - 1] + alpha * (
            self.river_centerline_y[idx] - self.river_centerline_y[idx - 1]
        )

        return float(x), float(y)

    def find_containing_triangle(self, x: float, y: float) -> int:
        """Find triangle containing point (x, y).

        Returns triangle index or -1 if outside mesh.
        """
        # Simple brute-force search (can be optimized with spatial index)
        for i in range(self.n_triangles):
            nodes = self.tri_nodes[i]
            x0, y0 = self.node_x[nodes[0]], self.node_y[nodes[0]]
            x1, y1 = self.node_x[nodes[1]], self.node_y[nodes[1]]
            x2, y2 = self.node_x[nodes[2]], self.node_y[nodes[2]]

            if _point_in_triangle(x, y, x0, y0, x1, y1, x2, y2):
                return i

        return -1


def _point_in_triangle(
    px: float,
    py: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> bool:
    """Check if point (px, py) is inside triangle using barycentric coordinates."""
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-10:
        return False

    a = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
    b = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
    c = 1 - a - b

    return (a >= 0) and (b >= 0) and (c >= 0)


def create_river_mesh(
    centerline_wgs84: list[tuple[float, float]] | None = None,
    width_m: float = 50.0,
    max_triangle_area: float = 500.0,
    depth_at_mouth: float = 3.0,
    depth_upstream: float = 1.0,
) -> Mesh:
    """Create a triangular mesh for the river domain.

    Args:
        centerline_wgs84: List of (lat, lon) points along river centerline.
                         Defaults to synthetic Waitangi River geometry.
        width_m: Average river width in meters.
        max_triangle_area: Maximum triangle area for mesh refinement (m²).
        depth_at_mouth: Water depth at river mouth (m).
        depth_upstream: Water depth at upstream extent (m).

    Returns:
        Mesh object with all geometry data.
    """
    if centerline_wgs84 is None:
        centerline_wgs84 = _default_waitangi_centerline()

    # Transform to NZTM2000
    transformer = Transformer.from_crs(
        WaitangiLocation.CRS_WGS84, WaitangiLocation.CRS_NZTM, always_xy=True
    )

    centerline_nztm = []
    for lat, lon in centerline_wgs84:
        x, y = transformer.transform(lon, lat)
        centerline_nztm.append((x, y))

    centerline_nztm = np.array(centerline_nztm)

    # Calculate chainage (cumulative distance)
    diffs = np.diff(centerline_nztm, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    chainage = np.concatenate([[0], np.cumsum(segment_lengths)])

    # Create river polygon by buffering centerline
    centerline_ls = LineString(centerline_nztm)
    river_polygon = centerline_ls.buffer(width_m / 2, cap_style="round")

    # Extract polygon boundary for triangulation
    boundary_coords = np.array(river_polygon.exterior.coords[:-1])  # Remove closing point

    # Mark segments for boundary conditions
    n_boundary = len(boundary_coords)

    # Create input for triangle library
    # Segments define boundary edges
    segments = [[i, (i + 1) % n_boundary] for i in range(n_boundary)]

    tri_input = {
        "vertices": boundary_coords,
        "segments": segments,
    }

    # Generate mesh with quality constraints
    # 'q' = quality mesh, 'a' = max area
    tri_output = triangle.triangulate(tri_input, f"pq30a{max_triangle_area}")

    vertices = tri_output["vertices"]
    triangles = tri_output["triangles"]

    # Calculate node depths (linear interpolation along river)
    node_depths = np.zeros(len(vertices))
    for i, (x, y) in enumerate(vertices):
        # Find nearest point on centerline
        point = Point(x, y)
        dist_along = centerline_ls.project(point)
        normalized_dist = dist_along / centerline_ls.length
        # Linear depth interpolation
        node_depths[i] = depth_at_mouth + normalized_dist * (depth_upstream - depth_at_mouth)

    # Calculate triangle centroids and areas
    centroids_x = np.mean(vertices[triangles, 0], axis=1)
    centroids_y = np.mean(vertices[triangles, 1], axis=1)

    # Triangle areas using cross product
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    areas = 0.5 * np.abs(
        (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
        - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
    )

    # Identify boundary nodes
    mouth_point = Point(centerline_nztm[0])
    upstream_point = Point(centerline_nztm[-1])

    mouth_indices = []
    upstream_indices = []
    boundary_threshold = width_m * 1.5

    for i, (x, y) in enumerate(vertices):
        p = Point(x, y)
        if p.distance(mouth_point) < boundary_threshold:
            mouth_indices.append(i)
        elif p.distance(upstream_point) < boundary_threshold:
            upstream_indices.append(i)

    return Mesh(
        node_x=jnp.asarray(vertices[:, 0], dtype=jnp.float32),
        node_y=jnp.asarray(vertices[:, 1], dtype=jnp.float32),
        node_depth=jnp.asarray(node_depths, dtype=jnp.float32),
        tri_nodes=jnp.asarray(triangles, dtype=jnp.int32),
        tri_centroids_x=jnp.asarray(centroids_x, dtype=jnp.float32),
        tri_centroids_y=jnp.asarray(centroids_y, dtype=jnp.float32),
        tri_areas=jnp.asarray(areas, dtype=jnp.float32),
        river_centerline_x=jnp.asarray(centerline_nztm[:, 0], dtype=jnp.float32),
        river_centerline_y=jnp.asarray(centerline_nztm[:, 1], dtype=jnp.float32),
        river_chainage=jnp.asarray(chainage, dtype=jnp.float32),
        mouth_node_indices=jnp.asarray(mouth_indices, dtype=jnp.int32),
        upstream_node_indices=jnp.asarray(upstream_indices, dtype=jnp.int32),
    )


def load_mesh_from_geojson(geojson_path: Path, **kwargs) -> Mesh:
    """Load mesh from a GeoJSON file containing river centerline.

    The GeoJSON should contain a LineString feature representing
    the river centerline from mouth to upstream.

    Args:
        geojson_path: Path to GeoJSON file.
        **kwargs: Additional arguments passed to create_river_mesh.

    Returns:
        Mesh object.
    """
    import json

    with open(geojson_path) as f:
        data = json.load(f)

    # Extract first LineString feature
    for feature in data.get("features", [data]):
        geom = feature.get("geometry", feature)
        if geom.get("type") == "LineString":
            coords = geom["coordinates"]
            # GeoJSON is [lon, lat], convert to (lat, lon)
            centerline = [(lat, lon) for lon, lat in coords]
            return create_river_mesh(centerline_wgs84=centerline, **kwargs)

    raise ValueError("No LineString geometry found in GeoJSON")


def _default_waitangi_centerline() -> list[tuple[float, float]]:
    """Get Waitangi River centerline from OpenStreetMap data.

    Returns points from river mouth upstream.
    Falls back to approximate coordinates if OSM fetch fails.
    """
    try:
        from waitangi.data.osm import fetch_waitangi_geometry_sync, get_cached_geometry

        # Try cache first
        data = get_cached_geometry()
        if data is None:
            data = fetch_waitangi_geometry_sync()

        centerline = data.get("river_centerline", [])
        if centerline:
            # OSM data is already (lat, lon) tuples
            return centerline

    except Exception as e:
        print(f"Warning: Could not fetch OSM data: {e}")

    # Fallback to approximate coordinates
    # Based on OSM data: river goes from ~174.08 (mouth) to ~173.82 (upstream)
    return [
        (-35.2711, 174.0797),  # Mouth - at Waitangi Bridge
        (-35.2719, 174.0796),  # Bridge
        (-35.2727, 174.0796),
        (-35.2774, 174.0504),
        (-35.2782, 174.0513),
        (-35.2868, 174.0257),
        (-35.2874, 174.0046),
        (-35.2993, 173.9617),
        (-35.3164, 173.8221),  # Upstream extent
    ]
