#!/usr/bin/env python3
"""Test access to Northland Regional Council Hilltop API for Waitangi River data.

The NRC provides hydrological data via Hilltop Server at:
http://hilltop.nrc.govt.nz/data.hts

Key sites for Waitangi River:
- Waitangi at Waimate North Rd - main flow/level gauge
- Waitangi at Wiroa Road - upstream gauge
"""

import requests
from urllib.parse import urlencode, quote
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta


BASE_URL = "http://hilltop.nrc.govt.nz/data.hts"


def hilltop_request(params: dict, timeout: int = 30) -> requests.Response:
    """Make a request to Hilltop API with proper URL encoding.

    Hilltop requires %20 for spaces, not + (which requests uses by default).
    """
    query_string = urlencode(params, quote_via=quote)
    url = f"{BASE_URL}?{query_string}"
    return requests.get(url, timeout=timeout)

# Primary monitoring site for Waitangi River flow
WAITANGI_SITE = "Waitangi at Waimate North Rd"


def test_site_list():
    """Test that we can retrieve the list of monitoring sites."""
    print("=== Testing SiteList request ===")

    response = hilltop_request({"Service": "Hilltop", "Request": "SiteList"})

    assert response.status_code == 200, f"Failed with status {response.status_code}"

    root = ET.fromstring(response.content)
    sites = [site.get("Name") for site in root.findall(".//Site")]

    waitangi_sites = [s for s in sites if "Waitangi" in s]
    print(f"Found {len(sites)} total sites")
    print(f"Waitangi sites: {waitangi_sites}")

    assert len(waitangi_sites) > 0, "No Waitangi sites found"
    assert WAITANGI_SITE in sites, f"{WAITANGI_SITE} not found in site list"

    print("✓ SiteList test passed\n")
    return sites


def test_measurement_list():
    """Test that we can retrieve available measurements for Waitangi site."""
    print(f"=== Testing MeasurementList for '{WAITANGI_SITE}' ===")

    response = hilltop_request({
        "Service": "Hilltop",
        "Request": "MeasurementList",
        "Site": WAITANGI_SITE,
    })

    assert response.status_code == 200, f"Failed with status {response.status_code}"

    root = ET.fromstring(response.content)

    # The XML structure has DataSource elements directly under HilltopServer
    # Each DataSource contains Measurement elements
    measurements = []
    for ds in root.findall("DataSource"):
        ds_name = ds.get("Name")
        for meas in ds.findall("Measurement"):
            meas_name = meas.get("Name")
            request_as = meas.find("RequestAs")
            request_as_text = request_as.text if request_as is not None else meas_name
            measurements.append({
                "datasource": ds_name,
                "name": meas_name,
                "request_as": request_as_text,
            })

    print(f"Found {len(measurements)} measurements:")
    for m in measurements[:10]:
        print(f"  - {m['datasource']}: {m['name']} (request as: {m['request_as']})")
    if len(measurements) > 10:
        print(f"  ... and {len(measurements) - 10} more")

    # Check for key measurements
    meas_names = [m["name"] for m in measurements]
    assert "Flow" in meas_names, "No Flow measurement found"
    assert "Stage" in meas_names, "No Stage measurement found"

    print("✓ MeasurementList test passed\n")
    return measurements


def test_get_recent_flow_data():
    """Test that we can retrieve actual flow data."""
    print(f"=== Testing GetData for Flow at '{WAITANGI_SITE}' ===")

    # Request last 7 days of data
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)

    response = hilltop_request({
        "Service": "Hilltop",
        "Request": "GetData",
        "Site": WAITANGI_SITE,
        "Measurement": "Flow",
        "From": start_time.strftime("%Y-%m-%d"),
        "To": end_time.strftime("%Y-%m-%d"),
    })

    assert response.status_code == 200, f"Failed with status {response.status_code}"

    root = ET.fromstring(response.content)

    # Parse measurement data
    data_points = []
    for measurement in root.findall(".//Measurement"):
        meas_name = measurement.get("Name")
        for data in measurement.findall(".//Data"):
            for value in data.findall("E"):
                timestamp = value.find("T").text if value.find("T") is not None else None
                val = value.find("I1").text if value.find("I1") is not None else None
                if timestamp and val:
                    data_points.append((timestamp, float(val)))

    print(f"Retrieved {len(data_points)} data points")

    if data_points:
        print(f"First reading: {data_points[0][0]} = {data_points[0][1]:.3f} m³/s")
        print(f"Last reading:  {data_points[-1][0]} = {data_points[-1][1]:.3f} m³/s")

        flows = [d[1] for d in data_points]
        print(f"Flow range: {min(flows):.3f} - {max(flows):.3f} m³/s")
        print(f"Mean flow: {sum(flows)/len(flows):.3f} m³/s")

    assert len(data_points) > 0, "No data points retrieved"

    print("✓ GetData (Flow) test passed\n")
    return data_points


def test_get_recent_stage_data():
    """Test that we can retrieve water level (stage) data."""
    print(f"=== Testing GetData for Stage at '{WAITANGI_SITE}' ===")

    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)

    response = hilltop_request({
        "Service": "Hilltop",
        "Request": "GetData",
        "Site": WAITANGI_SITE,
        "Measurement": "Stage",
        "From": start_time.strftime("%Y-%m-%d"),
        "To": end_time.strftime("%Y-%m-%d"),
    })

    assert response.status_code == 200, f"Failed with status {response.status_code}"

    root = ET.fromstring(response.content)

    data_points = []
    for measurement in root.findall(".//Measurement"):
        for data in measurement.findall(".//Data"):
            for value in data.findall("E"):
                timestamp = value.find("T").text if value.find("T") is not None else None
                val = value.find("I1").text if value.find("I1") is not None else None
                if timestamp and val:
                    data_points.append((timestamp, float(val)))

    print(f"Retrieved {len(data_points)} data points")

    if data_points:
        print(f"First reading: {data_points[0][0]} = {data_points[0][1]:.0f} mm")
        print(f"Last reading:  {data_points[-1][0]} = {data_points[-1][1]:.0f} mm")

        stages = [d[1] for d in data_points]
        print(f"Stage range: {min(stages):.0f} - {max(stages):.0f} mm")

    assert len(data_points) > 0, "No data points retrieved"

    print("✓ GetData (Stage) test passed\n")
    return data_points


def main():
    """Run all API tests."""
    print("=" * 60)
    print("Northland Regional Council Hilltop API Test")
    print(f"Base URL: {BASE_URL}")
    print(f"Target site: {WAITANGI_SITE}")
    print("=" * 60 + "\n")

    try:
        test_site_list()
        test_measurement_list()
        test_get_recent_flow_data()
        test_get_recent_stage_data()

        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except requests.RequestException as e:
        print(f"\n❌ NETWORK ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
