"""Debug visualization for estuary geometry."""

import matplotlib
matplotlib.use("macosx")

import matplotlib.pyplot as plt
from pyproj import Transformer
from shapely.geometry import MultiPolygon

from waitangi.core.config import WaitangiLocation
from waitangi.data.estuary_geometry import (
    get_estuary_polygon_nztm,
    get_linz_coastlines_nztm,
    get_navigable_centerline_nztm,
)
from waitangi.data.reference_points import (
    LANDMARKS,
    LAND_POINTS,
    WATER_POINTS_EAST,
    WATER_POINTS_WEST,
)


def to_nztm(lat: float, lon: float) -> tuple[float, float]:
    """Convert WGS84 to NZTM."""
    transformer = Transformer.from_crs(
        WaitangiLocation.CRS_WGS84, WaitangiLocation.CRS_NZTM, always_xy=True
    )
    return transformer.transform(lon, lat)


def plot_geometry_debug():
    """Create debug visualization of estuary geometry."""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor("#2d4a2d")  # Land green

    # Draw LINZ coastlines (raw data)
    print("Loading coastlines...")
    coastlines = get_linz_coastlines_nztm()
    for geom in coastlines:
        if geom.geom_type == "LineString":
            x, y = geom.xy
            ax.plot(x, y, "k-", linewidth=0.5, alpha=0.5)
        elif geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                x, y = line.xy
                ax.plot(x, y, "k-", linewidth=0.5, alpha=0.5)

    # Draw computed water polygon
    print("Loading water polygon...")
    estuary = get_estuary_polygon_nztm()

    if isinstance(estuary, MultiPolygon):
        for poly in estuary.geoms:
            x, y = poly.exterior.xy
            ax.fill(x, y, color="#1a5f8a", alpha=0.6, zorder=2)
            ax.plot(x, y, "b-", linewidth=1, zorder=3)
            # Draw holes
            for interior in poly.interiors:
                ix, iy = interior.xy
                ax.fill(ix, iy, color="#2d4a2d", zorder=2)
                ax.plot(ix, iy, "b-", linewidth=1, zorder=3)
    else:
        x, y = estuary.exterior.xy
        ax.fill(x, y, color="#1a5f8a", alpha=0.6, zorder=2)
        ax.plot(x, y, "b-", linewidth=1, zorder=3)

    # Draw centerline
    centerline = get_navigable_centerline_nztm()
    cx, cy = centerline.xy
    ax.plot(cx, cy, "c--", linewidth=2, label="Centerline", zorder=5)

    # Water points west (estuary)
    for pt in WATER_POINTS_WEST:
        x, y = to_nztm(pt.lat, pt.lon)
        ax.plot(x, y, "o", color="cyan", markersize=10, markeredgecolor="white", zorder=10)

    # Water points east (bay)
    for pt in WATER_POINTS_EAST:
        x, y = to_nztm(pt.lat, pt.lon)
        ax.plot(x, y, "s", color="deepskyblue", markersize=10, markeredgecolor="white", zorder=10)

    # Land points
    for pt in LAND_POINTS:
        x, y = to_nztm(pt.lat, pt.lon)
        # Check if point is incorrectly in water
        from shapely.geometry import Point
        pt_geom = Point(x, y)
        if estuary.contains(pt_geom):
            color = "orange"  # Incorrectly in water
        else:
            color = "red"
        ax.plot(x, y, "^", color=color, markersize=10, markeredgecolor="white", zorder=10)

    # Landmarks
    for pt in LANDMARKS:
        x, y = to_nztm(pt.lat, pt.lon)
        ax.plot(x, y, "*", color="yellow", markersize=15, markeredgecolor="black", zorder=11)
        ax.annotate(
            pt.name,
            (x, y),
            xytext=(8, 5),
            textcoords="offset points",
            color="white",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
        )

    ax.set_aspect("equal")
    ax.set_title(
        "Waitangi Estuary - LINZ Coastline Debug\n"
        "Blue=water, Cyan/DeepSkyBlue=water refs, Red=land refs (orange=in water incorrectly)",
        color="white",
        fontsize=12,
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    fig.patch.set_facecolor("#1a1a1a")
    plt.tight_layout()

    output_path = "estuary_debug_linz.png"
    plt.savefig(output_path, dpi=150, facecolor="#1a1a1a")
    print(f"Saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    plot_geometry_debug()
