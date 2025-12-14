"""Test all API keys to verify they work correctly."""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")


def test_linz():
    """Test LINZ Data Service API with a WFS GetCapabilities request."""
    print("Testing LINZ Data Service API...")

    from waitangi.core.config import get_settings

    settings = get_settings()
    api_key = settings.data_sources.linz_api_key

    if not api_key or api_key == "your_linz_api_key_here":
        print("  LINZ_API_KEY not set")
        return False

    import httpx

    wfs_url = f"{settings.data_sources.linz_base_url}/services;key={api_key}/wfs"
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetCapabilities",
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(wfs_url, params=params)

            if response.status_code == 200 and "WFS_Capabilities" in response.text:
                print(f"  LINZ API working! (key: {api_key[:8]}...)")
                return True
            elif response.status_code == 403:
                print(f"  LINZ API key invalid or expired")
                return False
            else:
                print(f"  LINZ API error: HTTP {response.status_code}")
                return False

    except Exception as e:
        print(f"  LINZ API error: {e}")
        return False


def test_metservice():
    """Test MetService/MetOcean API with a models list request."""
    print("Testing MetService/MetOcean API...")

    from waitangi.core.config import get_settings

    settings = get_settings()
    api_key = settings.data_sources.metservice_api_key

    if not api_key or api_key == "your_metservice_api_key_here":
        print("  METSERVICE_API_KEY not set (optional for now)")
        return True  # Optional, so don't fail

    import httpx

    # Test with models endpoint
    url = f"{settings.data_sources.metservice_base_url}/models/"
    headers = {
        "x-api-key": api_key,
        "accept": "application/json",
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                model_count = len(data) if isinstance(data, list) else "unknown"
                print(f"  MetService API working! ({model_count} models available)")
                return True
            elif response.status_code == 401:
                print("  MetService API key invalid")
                return False
            elif response.status_code == 403:
                print("  MetService API key forbidden")
                return False
            else:
                print(f"  MetService API error: HTTP {response.status_code}")
                print(f"     Response: {response.text[:200]}")
                return False

    except Exception as e:
        print(f"  MetService API error: {e}")
        return False


def test_linz_data_fetch():
    """Test fetching actual coastline data from LINZ."""
    print("Testing LINZ data fetch (MHW coastline layer)...")

    from waitangi.core.config import get_settings

    settings = get_settings()
    api_key = settings.data_sources.linz_api_key

    if not api_key or api_key == "your_linz_api_key_here":
        print("  Skipping (no API key)")
        return True

    from waitangi.data.linz import LINZClient

    try:
        client = LINZClient()
        # Fetch MHW coastline (should get ~119 features for Waitangi area)
        data = client.fetch_coastline_mhw()
        features = data.get("features", [])
        print(f"  LINZ data fetch working! Got {len(features)} coastline features")
        if features:
            geom_type = features[0].get("geometry", {}).get("type", "unknown")
            print(f"  Geometry type: {geom_type}")
        return len(features) > 0

    except Exception as e:
        print(f"  LINZ data fetch error: {e}")
        return False


def main():
    print("=" * 60)
    print("WAITANGI WATER MODELS - API KEY VERIFICATION")
    print("=" * 60)
    print()

    results = {
        "LINZ Auth": test_linz(),
        "LINZ Data": test_linz_data_fetch(),
        "MetService": test_metservice(),
    }

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed and name != "MetService":  # MetService is optional
            all_passed = False

    print()
    if all_passed:
        print("All required APIs are working!")
        return 0
    else:
        print("Some APIs failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
