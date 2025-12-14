"""LINZ Data Service client for high-resolution coastline and water body data.

LINZ (Land Information New Zealand) provides authoritative, detailed
geographic data for New Zealand including coastlines, rivers, and water bodies.

Requires a free API key from https://data.linz.govt.nz/
"""

import json
from pathlib import Path
from typing import Any

import httpx
from pyproj import Transformer

from waitangi.core.config import WaitangiLocation, get_settings


# LINZ layer IDs for relevant data
# Data is served in NZTM (EPSG:2193) coordinates
LINZ_LAYERS = {
    # NZ Coastline - Mean High Water (most accurate coastline)
    "coastline_mhw": 105085,
    # NZ Coastlines (Topo, 1:50k)
    "coastline_topo": 50258,
    # NZ Lake Polygons (Topo, 1:50k)
    "lakes": 50293,
    # NZ River Polygons (Topo, 1:50k) - river/estuary water bodies
    "river_polygons": 50327,
    # NZ River Centrelines (Topo, 1:50k)
    "river_centrelines": 50328,
    # NZ Coastlines and Islands Polygons (Topo, 1:50k) - land polygons
    "land_polygons": 51153,
}

# Bounding box for Waitangi area in WGS84 (min_lon, min_lat, max_lon, max_lat)
WAITANGI_BBOX_WGS84 = (174.04, -35.32, 174.13, -35.24)


def wgs84_bbox_to_nztm(bbox_wgs84: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Convert WGS84 bbox to NZTM bbox for LINZ queries."""
    transformer = Transformer.from_crs(WaitangiLocation.CRS_WGS84, WaitangiLocation.CRS_NZTM, always_xy=True)
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    min_e, min_n = transformer.transform(min_lon, min_lat)
    max_e, max_n = transformer.transform(max_lon, max_lat)
    return (min_e, min_n, max_e, max_n)


class LINZClient:
    """Client for fetching data from LINZ Data Service."""

    def __init__(self, api_key: str | None = None):
        settings = get_settings()
        self.api_key = api_key or settings.data_sources.linz_api_key
        self.base_url = str(settings.data_sources.linz_base_url)
        self.cache_dir = settings.data_sources.cache_dir / "linz"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.api_key:
            raise ValueError(
                "LINZ API key required. Get one free at https://data.linz.govt.nz/ "
                "and set DATA__LINZ_API_KEY in .env"
            )

    def _get_wfs_url(self, layer_id: int) -> str:
        """Get WFS URL for a layer."""
        return f"{self.base_url}/services;key={self.api_key}/wfs"

    def _cache_path(self, name: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{name}.geojson"

    def fetch_layer(
        self,
        layer_id: int,
        bbox_wgs84: tuple[float, float, float, float] = WAITANGI_BBOX_WGS84,
        use_cache: bool = True,
        cache_name: str | None = None,
    ) -> dict[str, Any]:
        """Fetch a layer from LINZ WFS service.

        Args:
            layer_id: LINZ layer ID
            bbox_wgs84: Bounding box in WGS84 (min_lon, min_lat, max_lon, max_lat)
            use_cache: Whether to use cached data if available
            cache_name: Name for cache file (defaults to layer_id)

        Returns:
            GeoJSON FeatureCollection (coordinates in NZTM)
        """
        cache_name = cache_name or str(layer_id)
        cache_file = self._cache_path(cache_name)

        if use_cache and cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)

        # Convert WGS84 bbox to NZTM for query
        bbox_nztm = wgs84_bbox_to_nztm(bbox_wgs84)

        # Fetch from WFS
        wfs_url = self._get_wfs_url(layer_id)
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": f"layer-{layer_id}",
            "outputFormat": "application/json",
            # NZTM bbox: minE,minN,maxE,maxN,CRS
            "bbox": f"{bbox_nztm[0]:.0f},{bbox_nztm[1]:.0f},{bbox_nztm[2]:.0f},{bbox_nztm[3]:.0f},EPSG:2193",
        }

        with httpx.Client(timeout=60.0) as client:
            response = client.get(wfs_url, params=params)
            response.raise_for_status()
            data = response.json()

        # Cache the result
        with open(cache_file, "w") as f:
            json.dump(data, f)

        return data

    def fetch_coastline_mhw(self, bbox_wgs84: tuple = WAITANGI_BBOX_WGS84) -> dict:
        """Fetch Mean High Water coastline data (most accurate)."""
        return self.fetch_layer(
            LINZ_LAYERS["coastline_mhw"],
            bbox_wgs84=bbox_wgs84,
            cache_name="coastline_mhw",
        )

    def fetch_coastline_topo(self, bbox_wgs84: tuple = WAITANGI_BBOX_WGS84) -> dict:
        """Fetch 1:50k topo coastline data."""
        return self.fetch_layer(
            LINZ_LAYERS["coastline_topo"],
            bbox_wgs84=bbox_wgs84,
            cache_name="coastline_topo",
        )

    def fetch_land_polygons(self, bbox_wgs84: tuple = WAITANGI_BBOX_WGS84) -> dict:
        """Fetch land/island polygons."""
        return self.fetch_layer(
            LINZ_LAYERS["land_polygons"],
            bbox_wgs84=bbox_wgs84,
            cache_name="land_polygons",
        )

    def fetch_river_polygons(self, bbox_wgs84: tuple = WAITANGI_BBOX_WGS84) -> dict:
        """Fetch river/estuary water body polygons."""
        return self.fetch_layer(
            LINZ_LAYERS["river_polygons"],
            bbox_wgs84=bbox_wgs84,
            cache_name="river_polygons",
        )

    def fetch_river_centrelines(self, bbox_wgs84: tuple = WAITANGI_BBOX_WGS84) -> dict:
        """Fetch river centreline data."""
        return self.fetch_layer(
            LINZ_LAYERS["river_centrelines"],
            bbox_wgs84=bbox_wgs84,
            cache_name="river_centrelines",
        )

    def fetch_all_waitangi_data(self) -> dict[str, dict]:
        """Fetch all relevant data for the Waitangi area.

        Returns:
            Dictionary with coastline and other geographic data (in NZTM coords).
        """
        print("Fetching LINZ data for Waitangi area...")

        data = {}

        print("  - Coastline (Mean High Water)...")
        data["coastline_mhw"] = self.fetch_coastline_mhw()
        print(f"    {len(data['coastline_mhw'].get('features', []))} features")

        print("  - Coastline (Topo 1:50k)...")
        data["coastline_topo"] = self.fetch_coastline_topo()
        print(f"    {len(data['coastline_topo'].get('features', []))} features")

        print("  - River polygons...")
        data["river_polygons"] = self.fetch_river_polygons()
        print(f"    {len(data['river_polygons'].get('features', []))} features")

        print("  - River centrelines...")
        data["river_centrelines"] = self.fetch_river_centrelines()
        print(f"    {len(data['river_centrelines'].get('features', []))} features")

        print("Done!")
        return data


def get_linz_client() -> LINZClient:
    """Get LINZ client instance."""
    return LINZClient()


def fetch_waitangi_linz_data() -> dict[str, dict]:
    """Fetch all LINZ data for Waitangi area."""
    client = get_linz_client()
    return client.fetch_all_waitangi_data()


if __name__ == "__main__":
    # Test fetching data
    try:
        data = fetch_waitangi_linz_data()
        print("\nData summary:")
        for key, geojson in data.items():
            features = geojson.get("features", [])
            print(f"  {key}: {len(features)} features")
    except ValueError as e:
        print(f"Error: {e}")
