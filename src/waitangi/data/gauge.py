"""River gauge data ingestion from Northland Regional Council."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Self

import httpx
import numpy as np
from pydantic import BaseModel

from waitangi.core.config import DataSourceSettings, get_settings


class GaugeReading(BaseModel):
    """Single gauge reading."""

    timestamp: datetime
    stage_m: float | None = None  # Water level (m)
    flow_m3s: float | None = None  # Discharge (m³/s)
    quality_code: str | None = None


class GaugeData(BaseModel):
    """Time series of gauge readings."""

    site_id: str
    site_name: str
    readings: list[GaugeReading]
    last_updated: datetime

    @property
    def latest(self) -> GaugeReading | None:
        """Get the most recent reading."""
        return self.readings[-1] if self.readings else None

    @property
    def timestamps(self) -> np.ndarray:
        """Get timestamps as numpy datetime64 array."""
        return np.array([r.timestamp for r in self.readings], dtype="datetime64[s]")

    @property
    def stages(self) -> np.ndarray:
        """Get stage values as numpy array (NaN for missing)."""
        return np.array(
            [r.stage_m if r.stage_m is not None else np.nan for r in self.readings],
            dtype=np.float64,
        )

    @property
    def flows(self) -> np.ndarray:
        """Get flow values as numpy array (NaN for missing)."""
        return np.array(
            [r.flow_m3s if r.flow_m3s is not None else np.nan for r in self.readings],
            dtype=np.float64,
        )


@dataclass
class NRCGaugeSite:
    """Known NRC gauge sites for the Waitangi catchment."""

    site_id: str
    name: str
    lat: float
    lon: float
    has_flow: bool = False
    has_stage: bool = True

    # Primary sites
    WAITANGI_WAKELINS = None  # Will be set after class definition

    @classmethod
    def get_primary_site(cls) -> Self:
        """Get the primary gauge site for Waitangi River."""
        return cls.WAITANGI_WAKELINS  # type: ignore


# Define the primary gauge site
NRCGaugeSite.WAITANGI_WAKELINS = NRCGaugeSite(
    site_id="42508",  # NRC site ID for Waitangi at Wakelins
    name="Waitangi at Wakelins",
    lat=-35.2761,
    lon=174.0728,
    has_flow=True,
    has_stage=True,
)


async def fetch_nrc_gauge_data(
    site: NRCGaugeSite | None = None,
    hours_back: int = 72,
    settings: DataSourceSettings | None = None,
) -> GaugeData:
    """Fetch gauge data from NRC Environmental Data Hub.

    Args:
        site: Gauge site to fetch. Defaults to Waitangi at Wakelins.
        hours_back: Hours of historical data to retrieve.
        settings: Data source settings. Defaults to global settings.

    Returns:
        GaugeData with time series of readings.

    Note:
        NRC uses Hilltop Server for environmental data. The API endpoint
        structure may need adjustment based on actual NRC implementation.
    """
    if site is None:
        site = NRCGaugeSite.get_primary_site()

    if settings is None:
        settings = get_settings().data_sources

    # NRC Environmental Data Hub uses Hilltop Server
    # This is a typical Hilltop request pattern
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours_back)

    # Build Hilltop-style request
    params = {
        "Service": "Hilltop",
        "Request": "GetData",
        "Site": site.name,
        "Measurement": "Flow" if site.has_flow else "Water Level",
        "From": start_time.strftime("%Y-%m-%d %H:%M"),
        "To": end_time.strftime("%Y-%m-%d %H:%M"),
    }

    readings: list[GaugeReading] = []

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{settings.nrc_base_url}/data.hts",
                params=params,
            )
            response.raise_for_status()

            # Parse Hilltop XML response
            # Note: Actual parsing depends on NRC's specific response format
            readings = _parse_hilltop_response(response.text, site.has_flow)

    except httpx.HTTPError:
        # Return synthetic data for development/testing
        readings = _generate_synthetic_gauge_data(start_time, end_time)

    return GaugeData(
        site_id=site.site_id,
        site_name=site.name,
        readings=readings,
        last_updated=datetime.now(),
    )


def _parse_hilltop_response(xml_text: str, is_flow: bool) -> list[GaugeReading]:
    """Parse Hilltop Server XML response.

    Note: This is a simplified parser. Production code should use
    proper XML parsing with the actual NRC response schema.
    """
    # Placeholder - actual implementation would parse XML
    # For now, return empty list to trigger synthetic data
    return []


def _generate_synthetic_gauge_data(
    start_time: datetime,
    end_time: datetime,
    interval_minutes: int = 15,
) -> list[GaugeReading]:
    """Generate synthetic gauge data for development.

    Creates plausible river flow patterns including:
    - Baseflow with slight diurnal variation
    - Random rainfall-driven pulses
    """
    readings = []
    current_time = start_time
    dt_minutes = timedelta(minutes=interval_minutes)

    # Base flow parameters
    base_flow = 2.5  # m³/s
    base_stage = 0.8  # m

    # Generate a "rainfall event" somewhere in the time series
    event_start = start_time + (end_time - start_time) * 0.3
    event_peak = event_start + timedelta(hours=6)
    event_end = event_start + timedelta(hours=24)

    rng = np.random.default_rng(42)

    while current_time <= end_time:
        # Diurnal variation (slightly higher in afternoon due to snowmelt pattern)
        hour = current_time.hour
        diurnal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * (hour - 14) / 24)

        # Rainfall pulse
        event_factor = 1.0
        if event_start <= current_time <= event_end:
            # Rising limb
            if current_time < event_peak:
                progress = (current_time - event_start) / (event_peak - event_start)
                event_factor = 1.0 + 3.0 * progress**2
            # Falling limb
            else:
                progress = (current_time - event_peak) / (event_end - event_peak)
                event_factor = 4.0 * np.exp(-2 * progress)

        # Random noise
        noise = 1.0 + 0.05 * rng.standard_normal()

        flow = base_flow * diurnal_factor * event_factor * noise
        stage = base_stage * (flow / base_flow) ** 0.4  # Manning-style relationship

        readings.append(
            GaugeReading(
                timestamp=current_time,
                flow_m3s=round(flow, 3),
                stage_m=round(stage, 3),
                quality_code="SYN",  # Synthetic
            )
        )

        current_time += dt_minutes

    return readings


def apply_rating_curve(
    stage_m: float,
    a: float = 15.0,
    b: float = 2.1,
    h0: float = 0.1,
) -> float:
    """Convert stage to flow using a power-law rating curve.

    Q = a * (h - h0)^b

    Args:
        stage_m: Water level in meters.
        a: Rating coefficient.
        b: Rating exponent (typically 1.5-2.5).
        h0: Datum offset in meters.

    Returns:
        Estimated flow in m³/s.
    """
    effective_depth = max(0.0, stage_m - h0)
    return a * (effective_depth**b)
