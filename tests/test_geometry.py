"""Tests for mesh geometry."""

import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from waitangi.models.geometry import Mesh, create_river_mesh


class TestMeshCreation:
    """Tests for mesh generation."""

    def test_create_default_mesh(self):
        """Should create mesh with default Waitangi geometry."""
        mesh = create_river_mesh()

        assert mesh.n_nodes > 0
        assert mesh.n_triangles > 0
        assert mesh.n_centerline_points > 0
        assert mesh.river_length > 0

    def test_mesh_has_valid_triangles(self):
        """All triangle nodes should reference valid nodes."""
        mesh = create_river_mesh()

        tri_nodes = np.asarray(mesh.tri_nodes)
        assert tri_nodes.min() >= 0
        assert tri_nodes.max() < mesh.n_nodes

    def test_mesh_has_boundary_nodes(self):
        """Should identify mouth and upstream boundary nodes."""
        mesh = create_river_mesh()

        assert len(mesh.mouth_node_indices) > 0
        assert len(mesh.upstream_node_indices) > 0

    def test_centerline_is_ordered(self):
        """Centerline should be ordered from mouth to upstream."""
        mesh = create_river_mesh()

        chainage = np.asarray(mesh.river_chainage)
        assert chainage[0] == 0
        assert np.all(np.diff(chainage) > 0)  # Monotonically increasing

    def test_depth_varies(self):
        """Depth should vary along river."""
        mesh = create_river_mesh()

        depths = np.asarray(mesh.node_depth)
        assert depths.min() > 0
        assert depths.max() > depths.min()

    def test_triangle_areas_positive(self):
        """All triangles should have positive area."""
        mesh = create_river_mesh()

        areas = np.asarray(mesh.tri_areas)
        assert np.all(areas > 0)

    def test_custom_width(self):
        """Should respect custom width parameter."""
        mesh_narrow = create_river_mesh(width_m=30.0)
        mesh_wide = create_river_mesh(width_m=80.0)

        # Wider mesh should have more nodes
        assert mesh_wide.n_nodes > mesh_narrow.n_nodes


class TestMeshOperations:
    """Tests for mesh query operations."""

    @pytest.fixture
    def mesh(self):
        return create_river_mesh()

    def test_point_to_chainage(self, mesh):
        """Should convert position to distance along river."""
        # Point at mouth
        mouth_x = float(mesh.river_centerline_x[0])
        mouth_y = float(mesh.river_centerline_y[0])
        chainage = mesh.point_to_chainage(mouth_x, mouth_y)
        assert chainage < 100  # Near mouth

        # Point upstream
        up_x = float(mesh.river_centerline_x[-1])
        up_y = float(mesh.river_centerline_y[-1])
        chainage_up = mesh.point_to_chainage(up_x, up_y)
        assert chainage_up > chainage

    def test_chainage_to_point(self, mesh):
        """Should convert chainage to position."""
        # At mouth
        x, y = mesh.chainage_to_point(0)
        assert abs(x - float(mesh.river_centerline_x[0])) < 50
        assert abs(y - float(mesh.river_centerline_y[0])) < 50

        # At end
        x, y = mesh.chainage_to_point(mesh.river_length)
        assert abs(x - float(mesh.river_centerline_x[-1])) < 50
        assert abs(y - float(mesh.river_centerline_y[-1])) < 50

    def test_round_trip_chainage(self, mesh):
        """chainage_to_point and point_to_chainage should be inverses."""
        for c in [200, 500, 1000, 2000]:
            if c > mesh.river_length:
                continue
            x, y = mesh.chainage_to_point(c)
            c_back = mesh.point_to_chainage(x, y)
            assert abs(c - c_back) < 150  # Within 150m (discretization error)

    def test_find_containing_triangle(self, mesh):
        """Should find triangle containing a point on river."""
        # Use a centerline point (should be inside mesh)
        mid_idx = mesh.n_centerline_points // 2
        x = float(mesh.river_centerline_x[mid_idx])
        y = float(mesh.river_centerline_y[mid_idx])

        tri_idx = mesh.find_containing_triangle(x, y)
        assert tri_idx >= 0


class TestMeshSerialization:
    """Tests for mesh save/load."""

    def test_save_and_load(self):
        """Should round-trip through NPZ file."""
        mesh = create_river_mesh()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mesh.npz"
            mesh.save(path)

            loaded = Mesh.load(path)

            assert loaded.n_nodes == mesh.n_nodes
            assert loaded.n_triangles == mesh.n_triangles
            assert loaded.river_length == mesh.river_length
            np.testing.assert_allclose(
                np.asarray(loaded.node_x),
                np.asarray(mesh.node_x),
            )

    def test_to_numpy(self):
        """Should export all arrays to numpy."""
        mesh = create_river_mesh()
        data = mesh.to_numpy()

        assert "node_x" in data
        assert "tri_nodes" in data
        assert isinstance(data["node_x"], np.ndarray)
