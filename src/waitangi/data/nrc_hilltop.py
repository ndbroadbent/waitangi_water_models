"""Northland Regional Council Hilltop API client for Waitangi River data.

Provides cached access to river flow and water level data from NRC's Hilltop server.
API calls are opt-in - by default, cached data is used.

Usage:
    client = NRCHilltopClient()

    # Use cached data (default)
    flow_data = client.get_flow_data(days=7)

    # Force refresh from API
    flow_data = client.get_flow_data(days=7, refresh=True)
"""

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote, urlencode

import requests


BASE_URL = "http://hilltop.nrc.govt.nz/data.hts"
WAITANGI_SITE = "Waitangi at Waimate North Rd"

# Cache directory
CACHE_DIR = Path(__file__).parent.parent.parent.parent / ".cache" / "nrc_hilltop"


@dataclass
class TimeSeriesPoint:
    """A single data point with timestamp and value."""

    timestamp: datetime
    value: float


@dataclass
class TimeSeries:
    """Time series data with metadata."""

    site: str
    measurement: str
    units: str
    points: list[TimeSeriesPoint]
    fetched_at: datetime
    from_cache: bool = False

    @property
    def values(self) -> list[float]:
        return [p.value for p in self.points]

    @property
    def timestamps(self) -> list[datetime]:
        return [p.timestamp for p in self.points]

    @property
    def min_value(self) -> float:
        return min(self.values) if self.values else 0.0

    @property
    def max_value(self) -> float:
        return max(self.values) if self.values else 0.0

    @property
    def mean_value(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    @property
    def latest(self) -> TimeSeriesPoint | None:
        return self.points[-1] if self.points else None


def _hilltop_request(params: dict, timeout: int = 30) -> requests.Response:
    """Make a request to Hilltop API with proper URL encoding.

    Hilltop requires %20 for spaces, not + (which requests uses by default).
    """
    query_string = urlencode(params, quote_via=quote)
    url = f"{BASE_URL}?{query_string}"
    return requests.get(url, timeout=timeout)


def _parse_time_series(xml_content: bytes, measurement: str) -> list[TimeSeriesPoint]:
    """Parse time series data from Hilltop XML response."""
    root = ET.fromstring(xml_content)

    points = []
    for meas_elem in root.findall(".//Measurement"):
        for data in meas_elem.findall(".//Data"):
            for value in data.findall("E"):
                timestamp_elem = value.find("T")
                value_elem = value.find("I1")
                if timestamp_elem is not None and value_elem is not None:
                    try:
                        ts = datetime.fromisoformat(timestamp_elem.text)
                        val = float(value_elem.text)
                        points.append(TimeSeriesPoint(timestamp=ts, value=val))
                    except (ValueError, TypeError):
                        continue

    return points


class NRCHilltopClient:
    """Client for NRC Hilltop API with caching support."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.site = WAITANGI_SITE

    def _cache_path(self, measurement: str, days: int) -> Path:
        """Get cache file path for a measurement."""
        return self.cache_dir / f"{measurement.lower()}_{days}d.json"

    def _load_cache(self, measurement: str, days: int) -> TimeSeries | None:
        """Load cached data if available and not too old."""
        cache_path = self._cache_path(measurement, days)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path) as f:
                data = json.load(f)

            # Check cache age - consider stale after 1 hour
            fetched_at = datetime.fromisoformat(data["fetched_at"])
            if datetime.now() - fetched_at > timedelta(hours=1):
                return None  # Cache is stale, but still usable if refresh=False

            points = [
                TimeSeriesPoint(
                    timestamp=datetime.fromisoformat(p["timestamp"]),
                    value=p["value"],
                )
                for p in data["points"]
            ]

            return TimeSeries(
                site=data["site"],
                measurement=data["measurement"],
                units=data["units"],
                points=points,
                fetched_at=fetched_at,
                from_cache=True,
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def _save_cache(self, series: TimeSeries, days: int) -> None:
        """Save time series to cache."""
        cache_path = self._cache_path(series.measurement, days)
        data = {
            "site": series.site,
            "measurement": series.measurement,
            "units": series.units,
            "fetched_at": series.fetched_at.isoformat(),
            "points": [
                {"timestamp": p.timestamp.isoformat(), "value": p.value}
                for p in series.points
            ],
        }
        with open(cache_path, "w") as f:
            json.dump(data, f)

    def _fetch_from_api(
        self, measurement: str, days: int, units: str
    ) -> TimeSeries:
        """Fetch data from NRC Hilltop API."""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        response = _hilltop_request({
            "Service": "Hilltop",
            "Request": "GetData",
            "Site": self.site,
            "Measurement": measurement,
            "From": start_time.strftime("%Y-%m-%d"),
            "To": end_time.strftime("%Y-%m-%d"),
        })

        if response.status_code != 200:
            raise RuntimeError(f"API request failed with status {response.status_code}")

        points = _parse_time_series(response.content, measurement)

        return TimeSeries(
            site=self.site,
            measurement=measurement,
            units=units,
            points=points,
            fetched_at=datetime.now(),
            from_cache=False,
        )

    def get_flow_data(self, days: int = 7, refresh: bool = False) -> TimeSeries:
        """Get river flow data (m³/s).

        Args:
            days: Number of days of historical data
            refresh: If True, fetch fresh data from API. If False, use cache if available.

        Returns:
            TimeSeries with flow data
        """
        if not refresh:
            cached = self._load_cache("Flow", days)
            if cached:
                return cached

        series = self._fetch_from_api("Flow", days, "m³/s")
        self._save_cache(series, days)
        return series

    def get_stage_data(self, days: int = 7, refresh: bool = False) -> TimeSeries:
        """Get water level/stage data (mm).

        Args:
            days: Number of days of historical data
            refresh: If True, fetch fresh data from API. If False, use cache if available.

        Returns:
            TimeSeries with stage data
        """
        if not refresh:
            cached = self._load_cache("Stage", days)
            if cached:
                return cached

        series = self._fetch_from_api("Stage", days, "mm")
        self._save_cache(series, days)
        return series

    def get_latest(self, refresh: bool = False) -> dict:
        """Get latest flow and stage readings.

        Args:
            refresh: If True, fetch fresh data from API.

        Returns:
            Dict with latest flow and stage values
        """
        flow = self.get_flow_data(days=1, refresh=refresh)
        stage = self.get_stage_data(days=1, refresh=refresh)

        return {
            "flow_m3s": flow.latest.value if flow.latest else None,
            "flow_timestamp": flow.latest.timestamp if flow.latest else None,
            "stage_mm": stage.latest.value if stage.latest else None,
            "stage_timestamp": stage.latest.timestamp if stage.latest else None,
            "from_cache": flow.from_cache and stage.from_cache,
        }

    def clear_cache(self) -> None:
        """Clear all cached data."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
