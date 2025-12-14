"""Geographic visualization and export tools."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from pyproj import Transformer

from waitangi.core.config import WaitangiLocation
from waitangi.models.geometry import Mesh
from waitangi.simulation.runner import SimulationResult


def create_river_map(
    mesh: Mesh,
    include_triangles: bool = False,
) -> dict:
    """Create a GeoJSON representation of the river geometry.

    Args:
        mesh: Mesh object with river geometry.
        include_triangles: Include mesh triangles in output.

    Returns:
        GeoJSON dictionary.
    """
    transformer = Transformer.from_crs(
        WaitangiLocation.CRS_NZTM, WaitangiLocation.CRS_WGS84, always_xy=True
    )

    features = []

    # Add centerline
    centerline_coords = []
    for i in range(mesh.n_centerline_points):
        x, y = float(mesh.river_centerline_x[i]), float(mesh.river_centerline_y[i])
        lon, lat = transformer.transform(x, y)
        centerline_coords.append([lon, lat])

    features.append({
        "type": "Feature",
        "properties": {
            "name": "River Centerline",
            "type": "centerline",
            "length_m": float(mesh.river_length),
        },
        "geometry": {
            "type": "LineString",
            "coordinates": centerline_coords,
        },
    })

    # Add mouth and upstream markers
    mouth_x = float(mesh.river_centerline_x[0])
    mouth_y = float(mesh.river_centerline_y[0])
    mouth_lon, mouth_lat = transformer.transform(mouth_x, mouth_y)

    features.append({
        "type": "Feature",
        "properties": {
            "name": "River Mouth",
            "type": "marker",
        },
        "geometry": {
            "type": "Point",
            "coordinates": [mouth_lon, mouth_lat],
        },
    })

    upstream_x = float(mesh.river_centerline_x[-1])
    upstream_y = float(mesh.river_centerline_y[-1])
    upstream_lon, upstream_lat = transformer.transform(upstream_x, upstream_y)

    features.append({
        "type": "Feature",
        "properties": {
            "name": "Upstream Extent",
            "type": "marker",
        },
        "geometry": {
            "type": "Point",
            "coordinates": [upstream_lon, upstream_lat],
        },
    })

    # Optionally add mesh triangles
    if include_triangles:
        node_x = np.asarray(mesh.node_x)
        node_y = np.asarray(mesh.node_y)
        tri_nodes = np.asarray(mesh.tri_nodes)

        for i in range(mesh.n_triangles):
            coords = []
            for j in range(3):
                idx = tri_nodes[i, j]
                x, y = node_x[idx], node_y[idx]
                lon, lat = transformer.transform(x, y)
                coords.append([lon, lat])
            coords.append(coords[0])  # Close polygon

            features.append({
                "type": "Feature",
                "properties": {
                    "type": "mesh_triangle",
                    "index": i,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords],
                },
            })

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def export_trajectory_geojson(
    result: SimulationResult,
    save_path: Path | None = None,
) -> dict:
    """Export simulation trajectory as GeoJSON.

    Args:
        result: Simulation result with trajectory data.
        save_path: Optional path to save GeoJSON file.

    Returns:
        GeoJSON dictionary.
    """
    transformer = Transformer.from_crs(
        WaitangiLocation.CRS_NZTM, WaitangiLocation.CRS_WGS84, always_xy=True
    )

    # Convert trajectory to WGS84
    coords = []
    for x, y in zip(result.trajectory_x, result.trajectory_y):
        lon, lat = transformer.transform(x, y)
        coords.append([lon, lat])

    # Create trajectory LineString
    trajectory_feature = {
        "type": "Feature",
        "properties": {
            "name": "Kayak Trajectory",
            "start_time": result.start_time.isoformat(),
            "end_time": result.end_time.isoformat(),
            "duration_seconds": result.duration_seconds,
            "total_distance_m": result.total_distance_m,
            "net_displacement_m": result.net_displacement_m,
            "mean_speed_ms": result.mean_speed_ms,
            "max_speed_ms": result.max_speed_ms,
            "tide_phase": result.tide_phase_at_start,
        },
        "geometry": {
            "type": "LineString",
            "coordinates": coords,
        },
    }

    # Add start and end points
    start_lon, start_lat = transformer.transform(
        result.trajectory_x[0], result.trajectory_y[0]
    )
    end_lon, end_lat = transformer.transform(
        result.trajectory_x[-1], result.trajectory_y[-1]
    )

    start_feature = {
        "type": "Feature",
        "properties": {
            "name": "Start",
            "time": result.start_time.isoformat(),
        },
        "geometry": {
            "type": "Point",
            "coordinates": [start_lon, start_lat],
        },
    }

    end_feature = {
        "type": "Feature",
        "properties": {
            "name": "End",
            "time": result.end_time.isoformat(),
        },
        "geometry": {
            "type": "Point",
            "coordinates": [end_lon, end_lat],
        },
    }

    geojson = {
        "type": "FeatureCollection",
        "features": [trajectory_feature, start_feature, end_feature],
    }

    if save_path:
        with open(save_path, "w") as f:
            json.dump(geojson, f, indent=2)

    return geojson


def export_particle_cloud_geojson(
    history: list,
    sample_interval: int = 10,
    save_path: Path | None = None,
) -> dict:
    """Export particle cloud evolution as GeoJSON.

    Args:
        history: List of ParticleState objects.
        sample_interval: Only export every N states.
        save_path: Optional path to save GeoJSON file.

    Returns:
        GeoJSON dictionary.
    """
    transformer = Transformer.from_crs(
        WaitangiLocation.CRS_NZTM, WaitangiLocation.CRS_WGS84, always_xy=True
    )

    features = []

    for i, state in enumerate(history[::sample_interval]):
        active_mask = np.asarray(state.active) > 0.5
        x = np.asarray(state.x)[active_mask]
        y = np.asarray(state.y)[active_mask]

        # Convert to WGS84
        coords = []
        for xi, yi in zip(x, y):
            lon, lat = transformer.transform(xi, yi)
            coords.append([lon, lat])

        # Create MultiPoint for this timestep
        features.append({
            "type": "Feature",
            "properties": {
                "time": state.timestamp.isoformat(),
                "timestep": i * sample_interval,
                "n_particles": len(coords),
            },
            "geometry": {
                "type": "MultiPoint",
                "coordinates": coords,
            },
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    if save_path:
        with open(save_path, "w") as f:
            json.dump(geojson, f, indent=2)

    return geojson


def create_velocity_field_geojson(
    velocity_field,
    mesh: Mesh,
    timestamp: datetime,
    sample_spacing_m: float = 100.0,
    save_path: Path | None = None,
) -> dict:
    """Create GeoJSON representation of velocity field.

    Args:
        velocity_field: VelocityField object.
        mesh: Mesh geometry.
        timestamp: Time for velocity calculation.
        sample_spacing_m: Spacing between velocity arrows.
        save_path: Optional path to save GeoJSON file.

    Returns:
        GeoJSON dictionary with velocity arrows as LineStrings.
    """
    transformer = Transformer.from_crs(
        WaitangiLocation.CRS_NZTM, WaitangiLocation.CRS_WGS84, always_xy=True
    )

    features = []

    # Sample along centerline
    n_samples = int(mesh.river_length / sample_spacing_m)
    chainages = np.linspace(0, mesh.river_length, n_samples)

    for chainage in chainages:
        x, y = mesh.chainage_to_point(chainage)
        u, v = velocity_field.get_velocity_at_point(x, y, timestamp)

        # Scale arrow length for visualization
        speed = np.sqrt(u**2 + v**2)
        scale = 50.0  # meters per m/s
        arrow_x = x + u * scale
        arrow_y = y + v * scale

        # Convert to WGS84
        start_lon, start_lat = transformer.transform(x, y)
        end_lon, end_lat = transformer.transform(arrow_x, arrow_y)

        features.append({
            "type": "Feature",
            "properties": {
                "type": "velocity_arrow",
                "chainage_m": chainage,
                "speed_ms": float(speed),
                "u_ms": float(u),
                "v_ms": float(v),
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[start_lon, start_lat], [end_lon, end_lat]],
            },
        })

    geojson = {
        "type": "FeatureCollection",
        "properties": {
            "timestamp": timestamp.isoformat(),
        },
        "features": features,
    }

    if save_path:
        with open(save_path, "w") as f:
            json.dump(geojson, f, indent=2)

    return geojson
